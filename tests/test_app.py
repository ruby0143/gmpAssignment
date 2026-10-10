import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import app  # noqa: E402


def make_df(n=6):
    return pd.DataFrame(
        {
            "review": [f"review {i}" for i in range(1, n + 1)],
            "date": pd.NaT,
            "rating": float("nan"),
            "sentiment": ["negative"] * n,
            "confidence": [0.9] * n,
            "low_confidence": [False] * n,
        }
    )


def good_payload():
    return {
        "pain_points": [
            {"theme": "Battery", "description": "d", "severity": 5, "review_ids": [1, 2, 3]},
            {"theme": "Sync", "description": "d", "severity": 3, "review_ids": [4, 99]},
            {"theme": "Ghost", "description": "d", "severity": 3, "review_ids": [99]},
        ],
        "recommendations": [
            {"title": "A", "rationale": "r", "impact": 5, "effort": 2, "pain_points": [1]},
            {"title": "B", "rationale": "r", "impact": 3, "effort": 3, "pain_points": [2]},
            {"title": "C", "rationale": "r", "impact": 4, "effort": 4, "pain_points": [1, 2]},
            {"title": "Ungrounded", "rationale": "r", "impact": 5, "effort": 1, "pain_points": [3]},
        ],
    }


def test_validate_drops_ungrounded_and_ranks():
    points, recs = app.validate_payload(good_payload(), 6)
    assert [p.theme for p in points] == ["Battery", "Sync"]
    assert points[1].review_ids == [4]  # invalid id 99 removed
    assert [r.title for r in recs] == ["A", "C", "B"]
    assert all(r.review_ids for r in recs)
    assert recs[0].score == round(5 * 3 / 2, 2)


def test_validate_rejects_too_few_recommendations():
    payload = good_payload()
    payload["recommendations"] = payload["recommendations"][:2]
    with pytest.raises(ValueError):
        app.validate_payload(payload, 6)


def test_validate_rejects_bad_rating():
    payload = good_payload()
    payload["pain_points"][0]["severity"] = 9
    with pytest.raises(ValueError):
        app.validate_payload(payload, 6)


def test_extract_json_with_fences():
    assert app._extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    with pytest.raises(ValueError):
        app._extract_json("no json")


def test_parse_text_and_limits():
    df, warnings = app.parse_reviews("good\n\nbad\n", None)
    assert list(df["review"]) == ["good", "bad"] and warnings
    with pytest.raises(app.AnalysisError):
        app.parse_reviews("  \n ", None)
    big, w = app.parse_reviews("\n".join(f"review {i}" for i in range(250)), None)
    assert len(big) == app.MAX_REVIEWS and w


def test_parse_csv_sample_and_bad_csv(tmp_path):
    df, _ = app.parse_reviews("", str(app.SAMPLE_PATH))
    assert len(df) == 24 and df["date"].notna().all()
    bad = tmp_path / "bad.csv"
    bad.write_text("foo,bar\n1,2\n")
    with pytest.raises(app.AnalysisError):
        app.parse_reviews("", str(bad))


def test_trend_requires_two_periods():
    df, _ = app.parse_reviews("", str(app.SAMPLE_PATH))
    df["sentiment"] = ["positive", "negative"] * 12
    trend = app.sentiment_trend(df)
    assert trend is not None and set(trend["sentiment"]) == set(app.SENTIMENTS)
    assert app.sentiment_trend(df.assign(date=pd.NaT)) is None


def test_missing_token_is_explicit(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(app.AnalysisError, match="HF_TOKEN"):
        app.call_llm("x")


def test_generate_retries_then_fails(monkeypatch):
    calls = []
    monkeypatch.setattr(app, "call_llm", lambda p: calls.append(1) or "not json")
    with pytest.raises(app.AnalysisError):
        app.generate_insights(make_df(), "My product", "")
    assert len(calls) == app.LLM_ATTEMPTS


def test_end_to_end_with_real_sentiment_model(monkeypatch):
    monkeypatch.setattr(app, "call_llm", lambda p: "Sure!\n" + json.dumps(good_payload()))
    out = app.analyze(
        "Battery is awful\nLove it\nSync fails\nWorks fine\nTerrible support\nOkay I guess",
        None, "Competitor product", "X", None, None, None, None, None, None, 200,
    )
    summary, dist, trend, pain, recs, table = out
    assert "6 reviews analyzed" in summary
    assert dist["reviews"].sum() == 6
    assert len(pain) == 2 and recs.count("###") == 3
    assert list(table.columns) == app.REVIEW_TABLE_COLUMNS


# ---------------------------------------------------------------- generic ingestion


def app_store_records(n=30):
    return [
        {
            "date": f"2020/{1 + i % 12}/{1 + i % 28}, 10:00:00",
            "userName": f"u{i}",
            "title": f"title {i}",
            "rating": 1 + i % 5,
            "review": f"body {i}",
            "app_name": "alpha" if i % 3 else "beta",
            "country": "at",
            "developerResponse": [{"body": "x"}] if i == 0 else None,
        }
        for i in range(n)
    ]


def test_json_app_store_schema_detection_and_normalization(tmp_path):
    f = tmp_path / "reviews.json"
    f.write_text(json.dumps(app_store_records()))
    raw = app.ingest.load_table(str(f))
    schema = app.ingest.detect_schema(raw)
    assert (schema.review, schema.title, schema.date, schema.rating, schema.group) == (
        "review", "title", "date", "rating", "app_name",
    )
    df, warnings = app.ingest.normalize(raw, schema)
    assert any("different products" in w for w in warnings)
    assert df.loc[0, "review"] == "title 0. body 0" and df["date"].notna().all() and df["rating"].between(1, 5).all()
    only_beta, _ = app.ingest.normalize(raw, schema, "beta")
    assert set(only_beta["group"]) == {"beta"}
    assert app.ingest.group_choices(raw, "app_name")[0] == ("alpha (20)", "alpha")


def test_unknown_column_names_fall_back_to_longest_text(tmp_path):
    f = tmp_path / "odd.csv"
    pd.DataFrame({"x1": ["a", "b", "c"], "x2": ["a long free text comment here", "another long opinion text", "third lengthy remark"]}).to_csv(f, index=False)
    assert app.ingest.detect_schema(app.ingest.load_table(str(f))).review == "x2"


def test_jsonl_wrapped_dict_list_of_strings_and_zip(tmp_path):
    import zipfile

    (tmp_path / "a.jsonl").write_text('{"text": "one"}\n{"text": "two"}\n')
    assert list(app.ingest.load_table(str(tmp_path / "a.jsonl"))["text"]) == ["one", "two"]
    (tmp_path / "b.json").write_text(json.dumps({"meta": 1, "reviews": [{"comment": "hi"}]}))
    assert list(app.ingest.load_table(str(tmp_path / "b.json")).columns) == ["comment"]
    (tmp_path / "c.json").write_text(json.dumps(["nice", "bad"]))
    assert list(app.ingest.load_table(str(tmp_path / "c.json"))["review"]) == ["nice", "bad"]
    z = tmp_path / "d.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("data/reviews.json", json.dumps(app_store_records(5)))
    assert len(app.ingest.load_table(str(z))) == 5


def test_ingest_errors(tmp_path):
    (tmp_path / "e.zip").write_bytes(b"not a zip")
    with pytest.raises(app.ingest.IngestError):
        app.ingest.load_table(str(tmp_path / "e.zip"))
    (tmp_path / "f.json").write_text('{"a": 1}')
    with pytest.raises(app.ingest.IngestError):
        app.ingest.load_table(str(tmp_path / "f.json"))
    with pytest.raises(app.ingest.IngestError):
        app.ingest.normalize(pd.DataFrame({"a": [1]}), app.ingest.Schema())


def test_parse_reviews_with_group_and_sampling(tmp_path):
    f = tmp_path / "r.json"
    f.write_text(json.dumps(app_store_records(300)))
    df, warnings = app.parse_reviews("", str(f), None, "alpha", 50)
    assert len(df) == 50 and set(df["group"]) == {"alpha"} and any("sampled" in w for w in warnings)
    again, _ = app.parse_reviews("", str(f), None, "alpha", 50)
    assert df["review"].tolist() == again["review"].tolist()  # reproducible


def test_trend_uses_years_for_long_spans_and_skips_sparse_periods():
    dates = pd.to_datetime(["2015-01-01"] * 3 + ["2018-06-01"] * 3 + ["2021-01-01"])
    df = pd.DataFrame({"date": dates, "sentiment": ["positive", "negative"] * 3 + ["positive"]})
    trend = app.sentiment_trend(df)
    assert sorted(trend["period"].unique()) == ["2015", "2018"]  # 2021 has <3 reviews


def test_english_only_filter():
    english = ["Awesome", "Super App", "This app is so good and I love it", "Please fix the sync, it keeps failing", "😄😄"]
    foreign = ["Sehr gute App, erfüllt Ihren Zweck", "Très bonne application mais pas assez rapide", "这个应用很好", "Лучшее приложение", "Nicht kompatibel"]
    assert all(app.ingest.is_english(t) for t in english)
    assert not any(app.ingest.is_english(t) for t in foreign)
    df, warnings = app.ingest.filter_english(pd.DataFrame({"review": english + foreign}))
    assert len(df) == len(english) and "non-English" in warnings[0]
    with pytest.raises(app.ingest.IngestError):
        app.ingest.filter_english(pd.DataFrame({"review": foreign}))


def test_load_dotenv_does_not_override(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('# comment\nDOTENV_A="one"\nDOTENV_B=two\nbad line\n')
    monkeypatch.setenv("DOTENV_B", "keep")
    monkeypatch.delenv("DOTENV_A", raising=False)
    app.load_dotenv(env)
    assert app.os.environ["DOTENV_A"] == "one" and app.os.environ["DOTENV_B"] == "keep"
    monkeypatch.delenv("DOTENV_A")
    app.load_dotenv(tmp_path / "missing.env")  # no error

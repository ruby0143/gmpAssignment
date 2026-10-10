"""Generic review ingestion: load CSV/TSV/JSON/JSONL (optionally zipped), detect which columns
hold the review text, date, rating and product, and normalize everything into one schema:

    review (str) | date (datetime64, NaT if unknown) | rating (float, NaN if unknown) | group (str | NA)
"""

from __future__ import annotations

import io
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

SUPPORTED_SUFFIXES = (".csv", ".tsv", ".json", ".jsonl", ".ndjson")
UPLOAD_TYPES = [*SUPPORTED_SUFFIXES, ".zip"]
MAX_FILE_BYTES = 50 * 1024 * 1024
MAX_GROUP_CHOICES = 100

REVIEW_NAMES = ("review", "review_text", "reviewtext", "review_body", "text", "content", "comment", "comments", "feedback", "body", "message", "description")
TITLE_NAMES = ("title", "review_title", "headline", "summary", "subject")
DATE_NAMES = ("date", "review_date", "created_at", "created", "timestamp", "time", "reviewed_at", "posted_at", "updated_at")
RATING_NAMES = ("rating", "stars", "star_rating", "score", "overall", "review_rating")
GROUP_NAMES = ("app_name", "product", "product_name", "app", "item", "brand", "business", "asin", "model", "name")


class IngestError(Exception):
    """A user-facing data problem."""


@dataclass
class Schema:
    review: str | None = None
    title: str | None = None
    date: str | None = None
    rating: str | None = None
    group: str | None = None


# ---------------------------------------------------------------------- loading


def _read_zip(data: bytes) -> tuple[str, bytes]:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            members = [
                m for m in zf.infolist()
                if not m.is_dir() and Path(m.filename).suffix.lower() in SUPPORTED_SUFFIXES and "__MACOSX" not in m.filename
            ]
            if not members:
                raise IngestError(f"The zip contains no supported file ({', '.join(SUPPORTED_SUFFIXES)}).")
            member = max(members, key=lambda m: m.file_size)
            if member.file_size > MAX_FILE_BYTES:
                raise IngestError("The file inside the zip is too large (limit 50 MB).")
            return member.filename, zf.read(member)
    except zipfile.BadZipFile as exc:
        raise IngestError("That file is not a valid zip archive.") from exc


def _records_from_json(obj: object) -> pd.DataFrame:
    if isinstance(obj, dict):
        lists = [v for v in obj.values() if isinstance(v, list) and v and isinstance(v[0], dict)]
        if not lists:
            raise IngestError("JSON object does not contain a list of review records.")
        obj = max(lists, key=len)
    if not isinstance(obj, list) or not obj:
        raise IngestError("JSON must be a non-empty list of review records.")
    if all(isinstance(v, str) for v in obj):
        return pd.DataFrame({"review": obj})
    if not all(isinstance(v, dict) for v in obj):
        raise IngestError("JSON list must contain either strings or objects.")
    return pd.DataFrame(obj)


def load_table(path: str) -> pd.DataFrame:
    """Load a CSV/TSV/JSON/JSONL file (or a zip containing one) into a raw DataFrame."""
    p = Path(path)
    if p.stat().st_size > MAX_FILE_BYTES:
        raise IngestError("File is too large (limit 50 MB).")
    name, data = p.name, p.read_bytes()
    if p.suffix.lower() == ".zip":
        name, data = _read_zip(data)
    suffix = Path(name).suffix.lower()
    buffer = io.BytesIO(data)
    try:
        if suffix in (".csv", ".tsv"):
            df = pd.read_csv(buffer, sep="\t" if suffix == ".tsv" else ",", encoding_errors="replace")
        elif suffix == ".json":
            try:
                df = _records_from_json(json.loads(data.decode("utf-8", errors="replace")))
            except json.JSONDecodeError:
                df = pd.read_json(io.BytesIO(data), lines=True)  # JSON Lines saved with a .json extension
        elif suffix in (".jsonl", ".ndjson"):
            df = pd.read_json(buffer, lines=True)
        else:
            raise IngestError(f"Unsupported file type '{suffix}'. Use {', '.join(UPLOAD_TYPES)}.")
    except (pd.errors.ParserError, pd.errors.EmptyDataError, ValueError) as exc:
        if isinstance(exc, IngestError):
            raise
        raise IngestError(f"Could not read '{name}': {exc}") from exc
    if df.empty:
        raise IngestError(f"'{name}' contains no rows.")
    return df


# ----------------------------------------------------------------- schema logic


def _find(columns: list[str], names: tuple[str, ...], taken: set[str]) -> str | None:
    lowered = {c.strip().lower().replace(" ", "_"): c for c in columns if c not in taken}
    return next((lowered[n] for n in names if n in lowered), None)


def detect_schema(df: pd.DataFrame) -> Schema:
    """Guess which columns hold review, title, date, rating and product; the UI lets the user override."""
    columns = [str(c) for c in df.columns]
    df = df.set_axis(columns, axis=1)
    taken: set[str] = set()
    schema = Schema()
    schema.review = _find(columns, REVIEW_NAMES, taken)
    if schema.review is None:
        text_cols = [c for c in columns if df[c].map(lambda v: isinstance(v, str)).mean() > 0.8]
        if text_cols:
            schema.review = max(text_cols, key=lambda c: df[c].astype(str).str.len().mean())
    if schema.review:
        taken.add(schema.review)
    schema.title = _find(columns, TITLE_NAMES, taken)
    taken.update(filter(None, [schema.title]))
    schema.date = _find(columns, DATE_NAMES, taken)
    taken.update(filter(None, [schema.date]))
    schema.rating = _find(columns, RATING_NAMES, taken)
    taken.update(filter(None, [schema.rating]))
    schema.group = _find(columns, GROUP_NAMES, taken)
    return schema


def group_choices(df: pd.DataFrame, column: str | None) -> list[tuple[str, str]]:
    """(label, value) pairs for the product filter, largest group first."""
    if not column or column not in df.columns:
        return []
    counts = df[column].dropna().astype(str).str.strip().value_counts().head(MAX_GROUP_CHOICES)
    return [(f"{value} ({count:,})", value) for value, count in counts.items() if value]


# ---------------------------------------------------------------- normalization


def normalize(df: pd.DataFrame, schema: Schema, group_value: str | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Map a raw table into review/date/rating/group. Returns the frame and non-fatal warnings."""
    df = df.set_axis([str(c) for c in df.columns], axis=1)
    if not schema.review or schema.review not in df.columns:
        raise IngestError(
            "Could not find the review text column. Pick it in 'Data mapping'. "
            f"Columns found: {', '.join(map(str, df.columns))}."
        )
    warnings: list[str] = []

    if schema.group and schema.group in df.columns:
        groups = df[schema.group].astype("string").str.strip()
        if group_value:
            df = df[groups == group_value]
            if df.empty:
                raise IngestError(f"No reviews found for '{group_value}'.")
        elif groups.nunique() > 1:
            warnings.append(
                f"Reviews from {groups.nunique()} different products are mixed together; "
                "select one product to get meaningful insights."
            )

    def text_of(column: str | None) -> pd.Series:
        if not column or column not in df.columns:
            return pd.Series("", index=df.index)
        return df[column].astype("string").fillna("").str.strip()

    body, title = text_of(schema.review), text_of(schema.title)
    review = (title.where(title == "", title + ". ") + body).str.strip()

    out = pd.DataFrame(
        {
            "review": review.astype(str),
            "date": pd.to_datetime(df[schema.date], errors="coerce")
            if schema.date and schema.date in df.columns
            else pd.NaT,
            "rating": pd.to_numeric(df[schema.rating], errors="coerce")
            if schema.rating and schema.rating in df.columns
            else float("nan"),
            "group": df[schema.group].astype("string").str.strip()
            if schema.group and schema.group in df.columns
            else pd.NA,
        },
        index=df.index,
    )
    if schema.date and schema.date in df.columns:
        bad = int(out["date"].isna().sum())
        if bad == len(out):
            warnings.append(f"Column '{schema.date}' could not be read as dates; trend analysis is disabled.")
        elif bad:
            warnings.append(f"{bad} review(s) have a missing or unparseable date and are excluded from the trend.")

    before = len(out)
    out = out[out["review"].str.len() > 0]
    if len(out) < before:
        warnings.append(f"Skipped {before - len(out)} empty review(s).")
    before = len(out)
    out = out.drop_duplicates(subset=["review", "date"])
    if len(out) < before:
        warnings.append(f"Removed {before - len(out)} duplicate review(s).")
    if out.empty:
        raise IngestError("No non-empty reviews were found.")
    return out.reset_index(drop=True), warnings


def sample_reviews(df: pd.DataFrame, limit: int, seed: int = 0) -> tuple[pd.DataFrame, list[str]]:
    """Reproducible random sample (not 'first N', which would bias toward the oldest reviews)."""
    if len(df) <= limit:
        return df, []
    sampled = df.sample(n=limit, random_state=seed).sort_index().reset_index(drop=True)
    return sampled, [f"Randomly sampled {limit} of {len(df):,} reviews (reproducible) to keep analysis fast and affordable."]


# ------------------------------------------------------------- English-only filter

ENGLISH_WORDS = frozenset(
    "the and is it to of in that this for with was not but are you have its it's my very so just i'm good great love "
    "like too can they all be as on at me what when will would there their out up one get got has had if from or an do "
    "don't dont no yes please can't".split()
)
FOREIGN_WORDS = {
    "de": "und nicht ist ich der das sehr mit ein eine für auf zu den von es sind aber auch wie nur noch schon gut gute guter beste bester kann einfach schnell man wenn mehr endlich nicht kein keine wird".split(),
    "fr": "le la les des est une pas que qui et je il très mais pour avec dans sur ce cette sont plus tout bien".split(),
    "es": "el los las es una que y muy pero para por del se como más bueno buena esta está gusta mucho".split(),
    "it": "il lo gli è una che e molto ma per con di del non come più bello sono questo".split(),
    "pt": "o os as é uma que e muito mas para com não do da em um são você está".split(),
    "nl": "de het een en is niet ik dat van voor met zijn maar ook heel erg goed".split(),
}
AMBIGUOUS = {"die", "plus", "con", "son", "a", "are"}
FOREIGN_LOOKUP = {w for words in FOREIGN_WORDS.values() for w in words} - ENGLISH_WORDS - AMBIGUOUS
_WORD = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)?", re.UNICODE)
_LATIN = re.compile(r"[A-Za-zÀ-ɏ]")
_LETTER = re.compile(r"[^\W\d_]", re.UNICODE)


def is_english(text: str) -> bool:
    """Offline heuristic (no model): rejects non-Latin scripts and text with more foreign than English function words.

    Short or ambiguous text (e.g. "Awesome", emoji-only) is kept so valid English reviews aren't dropped.
    """
    letters = _LETTER.findall(text)
    if not letters:
        return True
    if len(_LATIN.findall(text)) / len(letters) < 0.7:
        return False
    words = [w.lower() for w in _WORD.findall(text)]
    foreign = sum(w in FOREIGN_LOOKUP for w in words)
    english = sum(w in ENGLISH_WORDS for w in words)
    return not (foreign > english and len(words) >= 2)


def filter_english(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    keep = df["review"].map(is_english)
    if keep.all():
        return df, []
    dropped = int((~keep).sum())
    if not keep.any():
        raise IngestError("None of the reviews look English. This tool currently supports English-language reviews only.")
    return df[keep].reset_index(drop=True), [
        f"Excluded {dropped:,} review(s) that appear to be non-English (English only is supported)."
    ]

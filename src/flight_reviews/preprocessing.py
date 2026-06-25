"""
Text preprocessing pipeline for British Airways review data.

Design decisions
----------------
* Pure functions: every cleaning step is a standalone function that
  takes and returns a string.  This makes unit testing trivial and
  lets you compose or swap steps without touching other logic.
* Pre-compiled regex: patterns are compiled once at module load rather
  than inside the hot path (``apply`` over 3 000+ rows).
* Copy semantics: ``clean_dataframe`` never mutates the caller's DataFrame,
  which avoids the subtle bugs that ``inplace=True`` can cause.
"""

import logging
import re

import pandas as pd

from flight_reviews.config import MIN_WORD_LENGTH, STOPWORDS, VERIFIED_TAG

logger = logging.getLogger(__name__)

# Compiled once at module load — avoids recompilation on every row call.
_RE_NON_WORD = re.compile(r"[^\w\s]")
_RE_EXTRA_SPACE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Atomic cleaning steps
# ---------------------------------------------------------------------------

def normalise_case(text: str) -> str:
    """Return a lowercased copy of *text*."""
    return text.lower()


def remove_special_characters(text: str) -> str:
    """
    Strip punctuation and collapse runs of whitespace to a single space.

    Underscores are preserved (they are word characters in ``\\w``).
    """
    cleaned = _RE_NON_WORD.sub(" ", text)
    return _RE_EXTRA_SPACE.sub(" ", cleaned).strip()


def remove_verified_tag(text: str) -> str:
    """
    Drop the lowercase ``'trip verified'`` prefix inserted by Skytrax.

    This tag appears at the start of every verified review and carries
    zero sentiment signal; including it inflates its word-frequency rank.
    The input is expected to already be lowercased.
    """
    tag = VERIFIED_TAG.lower()
    return text.replace(tag, "").strip()


# ---------------------------------------------------------------------------
# Composed pipeline
# ---------------------------------------------------------------------------

def clean_review(text: str) -> str:
    """
    Apply the full single-review cleaning pipeline.

    Steps (in order):
    1. Normalise to lowercase.
    2. Strip punctuation and collapse whitespace.
    3. Remove the Skytrax verified-review tag.

    Parameters
    ----------
    text : str
        Raw review string as scraped from Skytrax.

    Returns
    -------
    str
        Cleaned text ready for sentiment scoring and word-frequency analysis.
    """
    text = normalise_case(text)
    text = remove_special_characters(text)
    text = remove_verified_tag(text)
    return text


def clean_dataframe(
    df: pd.DataFrame,
    text_col: str = "reviews",
) -> pd.DataFrame:
    """
    Apply the cleaning pipeline to an entire review DataFrame.

    Steps performed:
    * Drops the ``Unnamed: 0`` CSV-index artefact if present.
    * Deduplicates on *text_col*.
    * Filters to verified reviews only (containing ``VERIFIED_TAG``).
    * Applies :func:`clean_review` row-wise.
    * Removes rows that are empty strings after cleaning.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame with at least a *text_col* column.
    text_col : str
        Name of the column containing review text.

    Returns
    -------
    pd.DataFrame
        Cleaned copy with a reset integer index.
    """
    df = df.copy()

    # Remove the CSV index artefact written when index=True (the default).
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
        logger.debug("Dropped 'Unnamed: 0' index artefact column.")

    n_before = len(df)
    df = df.drop_duplicates(subset=[text_col])
    logger.info("Dropped %d duplicate rows (%d → %d).", n_before - len(df), n_before, len(df))

    # Keep only Skytrax-verified reviews; unverified ones lack an
    # authenticity signal and may skew sentiment distributions.
    df = df[df[text_col].str.contains(VERIFIED_TAG, na=False)]
    logger.info("Retained %d verified reviews after filtering.", len(df))

    df[text_col] = df[text_col].apply(clean_review)

    # Drop rows that became empty strings after tag removal.
    df = df[df[text_col].str.strip() != ""]

    return df.reset_index(drop=True)

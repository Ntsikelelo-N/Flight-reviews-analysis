"""
Sentiment analysis and word-frequency helpers for British Airways reviews.

Design decisions
----------------
* TextBlob is used for sentiment polarity scoring (range: -1.0 to +1.0).
  It requires no labelled training data, which suits this exploratory
  project.  The thresholds used to bucket polarity into
  positive/neutral/negative are defined in :mod:`config` so they can be
  tuned without touching this file.
* Word-frequency counting is intentionally done without scikit-learn's
  CountVectorizer so the logic remains readable and unit-testable with
  plain strings.  A production implementation would use the vectorizer.
* Pure functions only — no global state, no I/O, no side effects.
  Callers decide where results go.
"""

import logging
from collections import Counter

import pandas as pd
from textblob import TextBlob

from flight_reviews.config import MIN_WORD_LENGTH, STOPWORDS, TOP_N_WORDS

logger = logging.getLogger(__name__)

# Polarity bucket thresholds
_POSITIVE_THRESHOLD = 0.05   # TextBlob polarity values above this → positive
_NEGATIVE_THRESHOLD = -0.05  # Below this → negative; between the two → neutral


# ---------------------------------------------------------------------------
# Sentiment scoring
# ---------------------------------------------------------------------------

def score_polarity(text: str) -> float:
    """
    Return the TextBlob polarity score for *text*.

    Polarity ranges from -1.0 (most negative) to +1.0 (most positive).
    """
    return TextBlob(text).sentiment.polarity


def classify_sentiment(polarity: float) -> str:
    """
    Map a TextBlob polarity score to a human-readable sentiment label.

    Parameters
    ----------
    polarity : float
        Value in [-1.0, 1.0] as returned by :func:`score_polarity`.

    Returns
    -------
    str
        One of ``"positive"``, ``"neutral"``, or ``"negative"``.
    """
    if polarity > _POSITIVE_THRESHOLD:
        return "positive"
    if polarity < _NEGATIVE_THRESHOLD:
        return "negative"
    return "neutral"


def add_sentiment_columns(
    df: pd.DataFrame,
    text_col: str = "reviews",
) -> pd.DataFrame:
    """
    Add ``polarity`` and ``sentiment`` columns to a cleaned review DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned review DataFrame (output of
        :func:`~flight_reviews.preprocessing.clean_dataframe`).
    text_col : str
        Column name containing the cleaned review text.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with two additional columns:
        ``polarity`` (float) and ``sentiment`` (str).
    """
    df = df.copy()
    df["polarity"] = df[text_col].apply(score_polarity)
    df["sentiment"] = df["polarity"].apply(classify_sentiment)

    dist = df["sentiment"].value_counts().to_dict()
    logger.info("Sentiment distribution: %s", dist)

    return df


# ---------------------------------------------------------------------------
# Word-frequency analysis
# ---------------------------------------------------------------------------

def _tokenise(text: str) -> list[str]:
    """
    Split *text* into tokens, removing stopwords and very short words.

    This is a deliberately minimal tokeniser:  no stemming, no lemmatisation.
    The goal is readable words in charts, not recall-maximised NLP.
    """
    return [
        word for word in text.split()
        if word not in STOPWORDS and len(word) >= MIN_WORD_LENGTH
    ]


def compute_word_frequencies(
    df: pd.DataFrame,
    text_col: str = "reviews",
    sentiment_filter: str | None = None,
    top_n: int = TOP_N_WORDS,
) -> pd.DataFrame:
    """
    Count token frequencies across all reviews, with optional sentiment filter.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame that contains *text_col* and optionally a ``sentiment``
        column (required when *sentiment_filter* is not ``None``).
    text_col : str
        Column containing tokenisable review text.
    sentiment_filter : str or None
        When provided, limits analysis to rows whose ``sentiment`` column
        matches this value (e.g. ``"positive"``).
    top_n : int
        How many of the most-frequent tokens to return.

    Returns
    -------
    pd.DataFrame
        Two-column DataFrame: ``word`` and ``count``, sorted descending.
    """
    subset = df.copy()

    if sentiment_filter is not None:
        subset = subset[subset["sentiment"] == sentiment_filter]
        logger.info(
            "Computing word frequencies for '%s' reviews (%d rows).",
            sentiment_filter, len(subset),
        )

    all_tokens: list[str] = []
    for review_text in subset[text_col]:
        all_tokens.extend(_tokenise(review_text))

    most_common = Counter(all_tokens).most_common(top_n)
    return pd.DataFrame(most_common, columns=["word", "count"])


def get_summary_stats(df: pd.DataFrame) -> dict:
    """
    Return a plain-dict summary of the analysed DataFrame.

    Useful for assertions in tests and for logging a single-line project summary.

    Returns
    -------
    dict
        Keys: ``n_reviews``, ``n_positive``, ``n_negative``, ``n_neutral``,
        ``mean_polarity``, ``pct_positive``.
    """
    total = len(df)
    counts = df["sentiment"].value_counts()

    return {
        "n_reviews": total,
        "n_positive": int(counts.get("positive", 0)),
        "n_negative": int(counts.get("negative", 0)),
        "n_neutral":  int(counts.get("neutral", 0)),
        "mean_polarity": round(float(df["polarity"].mean()), 4),
        "pct_positive": round(100 * counts.get("positive", 0) / total, 1) if total else 0.0,
    }

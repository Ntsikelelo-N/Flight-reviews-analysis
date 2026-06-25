"""
Unit tests for flight_reviews.analysis.

Coverage targets
----------------
* Polarity scoring returns float in the expected range.
* Sentiment classification maps thresholds correctly.
* add_sentiment_columns adds both required columns and doesn't mutate input.
* Word-frequency computation respects stopword filtering and top-n limit.
* Sentiment filter in compute_word_frequencies restricts rows correctly.
* get_summary_stats returns a dict with the expected keys and coherent values.
"""

import pandas as pd
import pytest

from flight_reviews.analysis import (
    add_sentiment_columns,
    classify_sentiment,
    compute_word_frequencies,
    get_summary_stats,
    score_polarity,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def scored_df():
    """Small DataFrame with polarity and sentiment columns pre-populated."""
    data = {
        "reviews": [
            "the flight was absolutely wonderful and the crew were fantastic",
            "terrible experience rude staff and lost luggage",
            "the flight was okay nothing special",
        ],
        "polarity": [0.6, -0.5, 0.02],
        "sentiment": ["positive", "negative", "neutral"],
    }
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# score_polarity
# ---------------------------------------------------------------------------

class TestScorePolarity:
    def test_positive_text_yields_positive_score(self):
        assert score_polarity("the flight was excellent and the crew were amazing") > 0

    def test_negative_text_yields_negative_score(self):
        assert score_polarity("terrible service and rude staff") < 0

    def test_neutral_text_near_zero(self):
        score = score_polarity("the flight departed from heathrow")
        assert -0.2 < score < 0.2

    def test_returns_float(self):
        assert isinstance(score_polarity("good flight"), float)

    def test_empty_string_returns_zero(self):
        assert score_polarity("") == 0.0

    def test_polarity_within_range(self):
        score = score_polarity("absolutely brilliant wonderful amazing")
        assert -1.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# classify_sentiment
# ---------------------------------------------------------------------------

class TestClassifySentiment:
    def test_high_polarity_is_positive(self):
        assert classify_sentiment(0.6) == "positive"

    def test_low_polarity_is_negative(self):
        assert classify_sentiment(-0.4) == "negative"

    def test_zero_is_neutral(self):
        assert classify_sentiment(0.0) == "neutral"

    def test_boundary_above_threshold_is_positive(self):
        assert classify_sentiment(0.051) == "positive"

    def test_boundary_below_threshold_is_negative(self):
        assert classify_sentiment(-0.051) == "negative"

    def test_just_inside_neutral_band(self):
        assert classify_sentiment(0.04) == "neutral"
        assert classify_sentiment(-0.04) == "neutral"


# ---------------------------------------------------------------------------
# add_sentiment_columns
# ---------------------------------------------------------------------------

class TestAddSentimentColumns:
    def test_polarity_column_added(self):
        df = pd.DataFrame({"reviews": ["great flight", "awful service"]})
        result = add_sentiment_columns(df)
        assert "polarity" in result.columns

    def test_sentiment_column_added(self):
        df = pd.DataFrame({"reviews": ["great flight", "awful service"]})
        result = add_sentiment_columns(df)
        assert "sentiment" in result.columns

    def test_sentiment_values_are_valid_labels(self):
        df = pd.DataFrame({
            "reviews": [
                "fantastic crew and great service",
                "terrible delay and rude staff",
                "flight was okay",
            ]
        })
        result = add_sentiment_columns(df)
        valid = {"positive", "negative", "neutral"}
        assert set(result["sentiment"].unique()).issubset(valid)

    def test_original_df_not_mutated(self):
        df = pd.DataFrame({"reviews": ["great flight"]})
        original_cols = list(df.columns)
        add_sentiment_columns(df)
        assert list(df.columns) == original_cols

    def test_polarity_dtype_is_float(self):
        df = pd.DataFrame({"reviews": ["good flight"]})
        result = add_sentiment_columns(df)
        assert result["polarity"].dtype == float


# ---------------------------------------------------------------------------
# compute_word_frequencies
# ---------------------------------------------------------------------------

class TestComputeWordFrequencies:
    def test_returns_dataframe_with_word_and_count_columns(self, scored_df):
        result = compute_word_frequencies(scored_df)
        assert set(result.columns) == {"word", "count"}

    def test_respects_top_n_limit(self, scored_df):
        result = compute_word_frequencies(scored_df, top_n=2)
        assert len(result) <= 2

    def test_sorted_descending_by_count(self, scored_df):
        result = compute_word_frequencies(scored_df)
        counts = result["count"].tolist()
        assert counts == sorted(counts, reverse=True)

    def test_sentiment_filter_restricts_rows(self, scored_df):
        all_result = compute_word_frequencies(scored_df)
        pos_result = compute_word_frequencies(scored_df, sentiment_filter="positive")
        assert len(pos_result) <= len(all_result)

    def test_stopwords_excluded(self, scored_df):
        result = compute_word_frequencies(scored_df)
        stopword_hits = result[result["word"].isin(["the", "and", "was"])]
        assert len(stopword_hits) == 0


# ---------------------------------------------------------------------------
# get_summary_stats
# ---------------------------------------------------------------------------

class TestGetSummaryStats:
    def test_returns_dict(self, scored_df):
        assert isinstance(get_summary_stats(scored_df), dict)

    def test_required_keys_present(self, scored_df):
        stats = get_summary_stats(scored_df)
        expected_keys = {
            "n_reviews", "n_positive", "n_negative",
            "n_neutral", "mean_polarity", "pct_positive",
        }
        assert expected_keys.issubset(stats.keys())

    def test_n_reviews_matches_dataframe_length(self, scored_df):
        stats = get_summary_stats(scored_df)
        assert stats["n_reviews"] == len(scored_df)

    def test_sentiment_counts_sum_to_total(self, scored_df):
        stats = get_summary_stats(scored_df)
        total = stats["n_positive"] + stats["n_negative"] + stats["n_neutral"]
        assert total == stats["n_reviews"]

    def test_pct_positive_is_percentage(self, scored_df):
        stats = get_summary_stats(scored_df)
        assert 0.0 <= stats["pct_positive"] <= 100.0

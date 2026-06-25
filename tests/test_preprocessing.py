"""
Unit tests for flight_reviews.preprocessing.

Coverage targets
----------------
* Each atomic cleaning step in isolation (normalise_case,
  remove_special_characters, remove_verified_tag).
* The composed clean_review pipeline.
* The clean_dataframe function: deduplication, verified-filter,
  empty-row removal, and preservation of valid rows.

Test data is constructed inline so these tests have zero I/O dependencies
and run in milliseconds.
"""

import pandas as pd
import pytest

from flight_reviews.preprocessing import (
    clean_dataframe,
    clean_review,
    normalise_case,
    remove_special_characters,
    remove_verified_tag,
)


# ---------------------------------------------------------------------------
# normalise_case
# ---------------------------------------------------------------------------

class TestNormaliseCase:
    def test_uppercased_text_is_lowered(self):
        assert normalise_case("GREAT FLIGHT") == "great flight"

    def test_already_lowercase_unchanged(self):
        assert normalise_case("good service") == "good service"

    def test_mixed_case(self):
        assert normalise_case("MixED CaSe") == "mixed case"

    def test_empty_string_returned_as_empty(self):
        assert normalise_case("") == ""


# ---------------------------------------------------------------------------
# remove_special_characters
# ---------------------------------------------------------------------------

class TestRemoveSpecialCharacters:
    def test_punctuation_is_stripped(self):
        result = remove_special_characters("good flight!")
        assert "!" not in result

    def test_multiple_spaces_collapsed(self):
        result = remove_special_characters("good   service")
        assert "  " not in result

    def test_hyphen_removed(self):
        result = remove_special_characters("well-run airline")
        assert "-" not in result

    def test_plain_text_unchanged_after_strip(self):
        result = remove_special_characters("good flight")
        assert result == "good flight"


# ---------------------------------------------------------------------------
# remove_verified_tag
# ---------------------------------------------------------------------------

class TestRemoveVerifiedTag:
    def test_tag_at_start_is_removed(self):
        raw = "trip verified | seat was comfortable"
        result = remove_verified_tag(raw)
        assert "trip verified" not in result

    def test_text_without_tag_unchanged(self):
        raw = "flight was delayed by two hours"
        assert remove_verified_tag(raw) == raw

    def test_result_is_stripped(self):
        raw = "trip verified | good food"
        result = remove_verified_tag(raw)
        assert not result.startswith(" ")


# ---------------------------------------------------------------------------
# clean_review (composed pipeline)
# ---------------------------------------------------------------------------

class TestCleanReview:
    def test_uppercase_and_punctuation_removed(self):
        result = clean_review("Trip Verified | GREAT service!!!")
        assert result == result.lower()
        assert "!" not in result

    def test_verified_tag_stripped_from_pipeline(self):
        result = clean_review("Trip Verified | The crew was excellent.")
        assert "trip verified" not in result

    def test_returns_non_empty_string_for_valid_input(self):
        result = clean_review("Trip Verified | good flight overall")
        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# clean_dataframe
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df():
    """Minimal DataFrame mimicking raw Skytrax output."""
    return pd.DataFrame({
        "reviews": [
            "Trip Verified | Great service and on time",
            "Trip Verified | Terrible experience, lost luggage",
            "Trip Verified | Great service and on time",   # Duplicate
            "No tag here — not a verified review",
            "Trip Verified | ",                            # Becomes empty after cleaning
        ]
    })


class TestCleanDataframe:
    def test_returns_dataframe(self, sample_df):
        result = clean_dataframe(sample_df)
        assert isinstance(result, pd.DataFrame)

    def test_duplicates_removed(self, sample_df):
        result = clean_dataframe(sample_df)
        assert result["reviews"].duplicated().sum() == 0

    def test_unverified_reviews_excluded(self, sample_df):
        result = clean_dataframe(sample_df)
        assert not any("no tag" in row for row in result["reviews"])

    def test_index_reset(self, sample_df):
        result = clean_dataframe(sample_df)
        assert list(result.index) == list(range(len(result)))

    def test_does_not_mutate_original(self, sample_df):
        original_len = len(sample_df)
        clean_dataframe(sample_df)
        assert len(sample_df) == original_len

    def test_unnamed_column_dropped(self):
        df_with_artefact = pd.DataFrame({
            "Unnamed: 0": [0, 1],
            "reviews": [
                "Trip Verified | Good flight",
                "Trip Verified | Bad service",
            ],
        })
        result = clean_dataframe(df_with_artefact)
        assert "Unnamed: 0" not in result.columns

"""
Central configuration for the flight-reviews-analysis project.

All magic numbers, file paths, and tuneable parameters live here.
Importing from this module keeps downstream code free of hard-coded values
and makes experimentation (e.g. changing TOP_N_WORDS) a single-line change.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
REPORTS_DIR = ROOT_DIR / "reports" / "figures"

# ---------------------------------------------------------------------------
# Scraping
# ---------------------------------------------------------------------------
BASE_URL = "https://www.airlinequality.com/airline-reviews/british-airways"
PAGE_SIZE = 100          # Reviews per Skytrax page (max the site allows)
MAX_PAGES = 40           # Skytrax typically has ~36 pages at pagesize=100
REQUEST_DELAY_SECS = 1   # Polite crawl delay between requests
REQUEST_TIMEOUT_SECS = 10

# Tag that Skytrax prepends to reviews that have been verified
VERIFIED_TAG = "Trip Verified"

# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------
# English stopwords (NLTK-derived) + domain-specific additions.
# Centralised here so tests and the analysis module share the same set.
STOPWORDS: frozenset[str] = frozenset({
    "the", "and", "was", "for", "with", "that", "not", "were", "they",
    "but", "this", "had", "have", "from", "very", "you", "there", "are",
    "our", "all", "one", "which", "their", "get", "after", "when", "out",
    "its", "been", "just", "would", "also", "into", "than", "him", "her",
    "his", "she", "did", "will", "has", "more", "some", "your", "then",
    "what", "other", "back", "them", "even", "like", "can", "don", "too",
    "much", "could", "over", "said", "who", "how", "was", "are", "use",
    "used", "two", "got", "sat", "way",
})

# Tokens shorter than this are excluded from word-frequency counts
MIN_WORD_LENGTH = 3

# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
FIGURE_DPI = 120
TOP_N_WORDS = 20

# Colour palette keyed by sentiment label for consistent cross-plot colouring
SENTIMENT_PALETTE: dict[str, str] = {
    "positive": "#A8D8A8",  # Soft green
    "negative": "#F4A3A3",  # Soft red / salmon
    "neutral":  "#B0C4DE",  # Steel blue
}

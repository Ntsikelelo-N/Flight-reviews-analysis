# ---------------------------------------------------------------------------
# flight-reviews-analysis — developer workflow commands
# ---------------------------------------------------------------------------

.PHONY: setup test test-fast scrape run notebook lint clean help

PYTHON  ?= python
PYTEST  ?= pytest
PIP     ?= pip

setup:
	$(PIP) install -e ".[dev]"
	$(PYTHON) -m textblob.download_corpora

test:
	$(PYTEST) tests/ -v --tb=short --cov=src/flight_reviews --cov-report=term-missing

test-fast:
	$(PYTEST) tests/ -q

scrape:
	$(PYTHON) -c "\
from flight_reviews.scraper import scrape_reviews, save_raw_reviews; \
import logging; logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s'); \
reviews = scrape_reviews(); \
save_raw_reviews(reviews); \
print(f'Done — {len(reviews)} reviews saved to data/raw/')"

run:
	$(PYTHON) -c "\
import logging, pandas as pd; \
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s'); \
from flight_reviews.config import DATA_RAW_DIR, DATA_PROCESSED_DIR; \
from flight_reviews.preprocessing import clean_dataframe; \
from flight_reviews.analysis import add_sentiment_columns, compute_word_frequencies, get_summary_stats; \
from flight_reviews.visualisation import plot_sentiment_distribution, plot_polarity_histogram, plot_word_frequencies; \
df = pd.read_csv(DATA_RAW_DIR / 'BA_reviews_raw.csv'); \
df = clean_dataframe(df); \
df = add_sentiment_columns(df); \
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True); \
df.to_csv(DATA_PROCESSED_DIR / 'BA_reviews_processed.csv', index=False); \
print(get_summary_stats(df)); \
plot_sentiment_distribution(df); \
plot_polarity_histogram(df); \
[plot_word_frequencies(compute_word_frequencies(df, sentiment_filter=s), sentiment=s) for s in ['positive','negative','neutral']]; \
print('All figures saved to reports/figures/')"

notebook:
	jupyter lab notebooks/

lint:
	flake8 src/ tests/ --max-line-length=100 --extend-ignore=E203,W503

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	find . -name ".DS_Store" -delete
	rm -rf reports/figures/*.png
	rm -rf .pytest_cache
	rm -rf src/*.egg-info

help:
	@echo ""
	@echo "  make setup      install package + dev deps"
	@echo "  make test       full pytest suite with coverage"
	@echo "  make test-fast  tests without coverage"
	@echo "  make scrape     collect reviews from Skytrax"
	@echo "  make run        preprocess → analyse → plot"
	@echo "  make notebook   launch JupyterLab"
	@echo "  make lint       flake8 code-style check"
	@echo "  make clean      remove generated artefacts"
	@echo ""

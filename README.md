# British Airways Review Sentiment Analysis

![CI](https://github.com/Ntsikelelo-N/Flight-reviews-analysis/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue)
![Tests](https://img.shields.io/badge/tests-62%20passing-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-94%25-green)

End-to-end NLP pipeline that scrapes ~3 600 British Airways passenger reviews
from Skytrax, cleans the text, scores sentiment with TextBlob, and surfaces
patterns through word-frequency analysis and visualisation.

Built as a portfolio project demonstrating production-standard Python packaging,
modular design, and automated testing.

---

## Key findings

| Metric | Value |
|---|---|
| Reviews analysed | ~2 850 verified reviews |
| Positive sentiment | ~50% |
| Negative sentiment | ~24% |
| Mean polarity score | +0.11 (weakly positive) |
| Top positive words | service, crew, good, seat, comfortable |
| Top negative words | delayed, cancelled, poor, staff, baggage |

---

## Project structure
Flight-reviews-analysis/

├── src/flight_reviews/    

│   ├── config.py           

│   ├── scraper.py          

│   ├── preprocessing.py    

│   ├── analysis.py         

│   └── visualisation.py    

├── tests/                  

│   ├── test_preprocessing.py

│   ├── test_analysis.py

│   └── test_scraper.py

├── notebooks/             

│   └── BA_reviews_analysis.ipynb

├── data/

│   ├── raw/                

│   └── processed/          

├── reports/figures/        

├── Makefile                

└── setup.py                
---

## Quickstart

```bash
git clone https://github.com/Ntsikelelo-N/Flight-reviews-analysis.git
cd Flight-reviews-analysis
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
make setup      # install package + dev dependencies
make test       # run 62 tests with coverage report
make scrape     # collect reviews from Skytrax (~50 seconds)
make run        # preprocess → score → generate all charts
make notebook   # open JupyterLab
```

---

## Makefile targets

| Target | What it does |
|---|---|
| `make setup` | `pip install -e ".[dev]"` + TextBlob corpora |
| `make test` | Full pytest suite with coverage |
| `make test-fast` | Tests without coverage (faster feedback) |
| `make scrape` | Collect raw reviews from Skytrax |
| `make run` | Full pipeline: preprocess → analyse → plot |
| `make notebook` | Launch JupyterLab |
| `make lint` | flake8 style check |
| `make clean` | Remove generated artefacts |

---

## Tech stack

| Layer | Tools |
|---|---|
| Scraping | `requests`, `BeautifulSoup4` |
| Data manipulation | `pandas` |
| NLP / Sentiment | `TextBlob` (lexicon-based polarity) |
| Visualisation | `matplotlib`, `seaborn` |
| Testing | `pytest`, `pytest-cov`, `unittest.mock` |
| CI | GitHub Actions (Python 3.10 + 3.11) |
| Packaging | `setuptools`, editable install |

---

## Author

**Ntsikelelo Nicholas Jantjie**
Junior Data Scientist | Johannesburg, South Africa
[GitHub](https://github.com/Ntsikelelo-N) · [LinkedIn](https://www.linkedin.com/in/ntsikelelo-jantjie) · ntsikelelonamba@gmail.com

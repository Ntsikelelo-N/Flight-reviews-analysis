"""
flight_reviews
==============
Sentiment analysis and EDA on British Airways passenger reviews
scraped from Skytrax.

Public surface area intentionally kept minimal — import from submodules
(scraper, preprocessing, analysis, visualisation) for specific functionality.
"""

try:
    from importlib.metadata import version, PackageNotFoundError

    __version__ = version("flight-reviews-analysis")
except Exception:
    __version__ = "dev"

__all__ = ["__version__"]

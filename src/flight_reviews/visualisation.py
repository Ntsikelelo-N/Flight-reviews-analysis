"""
Visualisation helpers for the British Airways sentiment analysis project.

Design decisions
----------------
* All plot code lives here, not in notebooks.  The notebook simply calls
  these functions and displays what they return.  This makes plots
  reproducible from the CLI, importable in tests, and easy to restyle
  centrally.
* Every function accepts an optional ``ax`` parameter so callers can
  compose multi-panel figures without duplication.
* Figures are saved to ``reports/figures/`` with descriptive names; the
  function returns the ``Figure`` so callers can display it in a notebook
  via ``plt.show()`` or save it to a different path if needed.
* The seaborn theme is applied once at module load — callers don't need
  to set it themselves.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns

from flight_reviews.config import FIGURE_DPI, REPORTS_DIR, SENTIMENT_PALETTE

logger = logging.getLogger(__name__)

# Apply a clean, publication-friendly theme once at module load.
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, filename: str) -> Path:
    """Save *fig* to ``reports/figures/`` and return the resolved path."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORTS_DIR / filename
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    logger.info("Figure saved → %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

def plot_sentiment_distribution(
    df: pd.DataFrame,
    save: bool = True,
    filename: str = "sentiment_distribution.png",
) -> plt.Figure:
    """
    Bar chart of positive / neutral / negative review counts.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a ``sentiment`` column.
    save : bool
        Whether to save the figure to ``reports/figures/``.
    filename : str
        Output filename.

    Returns
    -------
    matplotlib.figure.Figure
    """
    counts = df["sentiment"].value_counts().reindex(
        ["positive", "neutral", "negative"], fill_value=0
    )
    colours = [SENTIMENT_PALETTE.get(s, "#cccccc") for s in counts.index]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(counts.index, counts.values, color=colours, edgecolor="white", linewidth=0.8)

    # Annotate bar heights
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5,
            str(int(bar.get_height())),
            ha="center", va="bottom", fontsize=10,
        )

    ax.set_title("British Airways Review Sentiment Distribution", fontweight="bold", pad=12)
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Number of Reviews")
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    fig.tight_layout()

    if save:
        _save(fig, filename)
    return fig


def plot_polarity_histogram(
    df: pd.DataFrame,
    bins: int = 40,
    save: bool = True,
    filename: str = "polarity_histogram.png",
) -> plt.Figure:
    """
    Histogram of TextBlob polarity scores across all reviews.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a ``polarity`` column.
    bins : int
        Number of histogram bins.
    save : bool
        Whether to save the figure.
    filename : str
        Output filename.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df["polarity"], bins=bins, color="#B0C4DE", edgecolor="white", linewidth=0.6)
    ax.axvline(0, color="#888", linestyle="--", linewidth=1, label="Neutral boundary")
    ax.axvline(
        df["polarity"].mean(), color="#E07B54",
        linestyle="-", linewidth=1.5, label=f"Mean ({df['polarity'].mean():.3f})",
    )
    ax.set_title("Distribution of Polarity Scores", fontweight="bold", pad=12)
    ax.set_xlabel("Polarity Score  (−1 = most negative  |  +1 = most positive)")
    ax.set_ylabel("Count")
    ax.legend(frameon=False)
    fig.tight_layout()

    if save:
        _save(fig, filename)
    return fig


def plot_word_frequencies(
    word_freq_df: pd.DataFrame,
    sentiment: str = "all",
    save: bool = True,
) -> plt.Figure:
    """
    Horizontal bar chart of the most frequent words for a given sentiment.

    Parameters
    ----------
    word_freq_df : pd.DataFrame
        Two-column DataFrame (``word``, ``count``) as returned by
        :func:`~flight_reviews.analysis.compute_word_frequencies`.
    sentiment : str
        Used in the chart title and output filename.  Pass ``"positive"``,
        ``"negative"``, ``"neutral"``, or ``"all"``.
    save : bool
        Whether to save the figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    colour = SENTIMENT_PALETTE.get(sentiment, "#B0C4DE")

    fig, ax = plt.subplots(figsize=(8, max(4, len(word_freq_df) * 0.35)))
    ax.barh(
        word_freq_df["word"],
        word_freq_df["count"],
        color=colour, edgecolor="white", linewidth=0.6,
    )
    ax.invert_yaxis()   # Most frequent word at the top
    ax.set_title(
        f"Top {len(word_freq_df)} Words — {sentiment.capitalize()} Reviews",
        fontweight="bold", pad=12,
    )
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Word")
    fig.tight_layout()

    if save:
        filename = f"word_freq_{sentiment}.png"
        _save(fig, filename)
    return fig


def plot_polarity_by_recommendation(
    df: pd.DataFrame,
    save: bool = True,
    filename: str = "polarity_by_recommendation.png",
) -> plt.Figure:
    """
    Box plot of polarity scores split by whether the reviewer recommended BA.

    Only produced when a ``recommended`` column is present; returns ``None``
    otherwise so callers can guard on it without crashing.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``polarity`` and ``recommended`` columns.
    save : bool
        Whether to save the figure.
    filename : str
        Output filename.

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    if "recommended" not in df.columns:
        logger.warning(
            "Column 'recommended' not found — skipping recommendation box plot."
        )
        return None

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(
        data=df, x="recommended", y="polarity",
        palette={"yes": SENTIMENT_PALETTE["positive"], "no": SENTIMENT_PALETTE["negative"]},
        width=0.45, linewidth=1.2, ax=ax,
    )
    ax.axhline(0, color="#888", linestyle="--", linewidth=0.8)
    ax.set_title("Polarity Score by Recommendation", fontweight="bold", pad=12)
    ax.set_xlabel("Would Recommend British Airways?")
    ax.set_ylabel("TextBlob Polarity Score")
    fig.tight_layout()

    if save:
        _save(fig, filename)
    return fig

"""
Minimal setup.py for flight-reviews-analysis.

Using ``pip install -e .`` installs the package in editable mode so that
``import flight_reviews`` resolves to src/flight_reviews/ regardless of
the working directory — no ``sys.path`` hacks needed.
"""

from setuptools import find_packages, setup

setup(
    name="flight-reviews-analysis",
    version="0.1.0",
    author="Ntsikelelo Nicholas Jantjie",
    author_email="ntsikelelonamba@gmail.com",
    description=(
        "Sentiment analysis and EDA on British Airways passenger reviews "
        "scraped from Skytrax."
    ),
    url="https://github.com/Ntsikelelo-N/Flight-reviews-analysis",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        "beautifulsoup4>=4.12",
        "matplotlib>=3.9",
        "pandas>=2.2",
        "requests>=2.31",
        "scikit-learn>=1.6",
        "seaborn>=0.13",
        "textblob>=0.19",
    ],
    extras_require={
        "dev": [
            "ipykernel>=6.29",
            "jupyter>=1.0",
            "pytest>=8.0",
            "pytest-cov>=5.0",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

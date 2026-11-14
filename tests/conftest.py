"""Shared fixtures: tiny in-memory datasets and NLTK setup."""

import pandas as pd
import pytest

from disaster_response.etl import clean_data
from disaster_response.text import ensure_nltk_resources


@pytest.fixture(scope="session", autouse=True)
def _nltk_resources():
    # Downloads punkt/punkt_tab/wordnet/stopwords on first run (CI included);
    # no-op when they are already present.
    ensure_nltk_resources()


@pytest.fixture()
def raw_df():
    """Small raw frame as produced by merging the two CSVs.

    Includes a ``related-2`` value (clipped to 1 by cleaning) and a
    duplicated row (dropped by cleaning).
    """
    rows = [
        {
            "id": "1",
            "message": "Need water and food urgently",
            "original": "",
            "genre": "direct",
            "categories": "related-1;request-1;water-1;food-1",
        },
        {
            "id": "2",
            "message": "Storm destroyed the shelter",
            "original": "",
            "genre": "news",
            "categories": "related-2;request-0;water-0;food-0",
        },
        {
            "id": "3",
            "message": "Is anyone offering medical help?",
            "original": "",
            "genre": "social",
            "categories": "related-1;request-0;water-0;food-0",
        },
    ]
    df = pd.DataFrame(rows + [rows[0]])  # one exact duplicate
    return df


@pytest.fixture()
def cleaned_df(raw_df):
    return clean_data(raw_df)


def make_tiny_dataset(n_per_class: int = 14) -> pd.DataFrame:
    """Build a small labelled dataset for model/app smoke tests.

    Three categories (``water``, ``food``, ``shelter``) with keyword-rich
    messages, spread across the three genres.
    """
    templates = [
        ("We urgently need clean drinking water", "water", "direct"),
        ("No water left in the village, please help", "water", "news"),
        ("Water supplies are running low after the flood", "water", "social"),
        ("Families need food and rice supplies", "food", "direct"),
        ("Food distribution center is open today", "food", "news"),
        ("Children are hungry, we need food donations", "food", "social"),
        ("Our shelter was destroyed by the storm", "shelter", "direct"),
        ("Temporary shelter needed for displaced families", "shelter", "news"),
        ("The shelter roof collapsed during the earthquake", "shelter", "social"),
    ]
    rows = []
    for i in range(n_per_class):
        for j, (message, category, genre) in enumerate(templates):
            rows.append(
                {
                    "id": str(i * len(templates) + j),
                    "message": f"{message} (report {i})",
                    "original": "",
                    "genre": genre,
                    "related": 1,
                    "water": int(category == "water"),
                    "food": int(category == "food"),
                    "shelter": int(category == "shelter"),
                }
            )
    return pd.DataFrame(rows)

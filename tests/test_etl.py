"""Tests for the ETL step (load -> clean -> save)."""

import pandas as pd
import pytest
from sqlalchemy import create_engine

from disaster_response.etl import clean_data, load_data, save_data


def test_load_data_merges_on_id(tmp_path):
    messages = tmp_path / "messages.csv"
    categories = tmp_path / "categories.csv"
    messages.write_text(
        "id,message,original,genre\n"
        "1,Need water,,direct\n"
        "2,Storm coming,,news\n"
    )
    categories.write_text(
        "id,categories\n"
        "1,related-1;water-1\n"
        "2,related-1;water-0\n"
    )
    df = load_data(str(messages), str(categories))
    assert len(df) == 2
    assert list(df["id"]) == ["1", "2"]
    assert "categories" in df.columns


def test_clean_data_splits_categories(cleaned_df):
    for column in ["related", "request", "water", "food"]:
        assert column in cleaned_df.columns
    assert set(cleaned_df["water"].unique()) <= {0, 1}
    # First row: request-1, water-1, food-1.
    first = cleaned_df.iloc[0]
    assert (first["request"], first["water"], first["food"]) == (1, 1, 1)


def test_clean_data_clips_values_above_one(cleaned_df):
    # Row 2 has related-2 in the raw data; it must become 1, not 2.
    assert cleaned_df["related"].max() == 1
    assert cleaned_df.loc[cleaned_df["id"] == "2", "related"].iloc[0] == 1


def test_clean_data_drops_duplicates(cleaned_df):
    assert len(cleaned_df) == 3
    assert cleaned_df["id"].is_unique


def test_clean_data_drops_original_categories_column(cleaned_df):
    assert "categories" not in cleaned_df.columns


def test_save_and_load_roundtrip(cleaned_df, tmp_path):
    db_path = tmp_path / "Test.db"
    table_name = save_data(cleaned_df, str(db_path))
    assert table_name == "Test"
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        reloaded = pd.read_sql_table("Test", con=engine)
    finally:
        engine.dispose()
    assert len(reloaded) == len(cleaned_df)
    assert list(reloaded.columns) == list(cleaned_df.columns)
    assert reloaded["water"].tolist() == cleaned_df["water"].tolist()


def test_clean_data_requires_categories_column():
    with pytest.raises(KeyError):
        clean_data(pd.DataFrame({"message": ["hello"]}))

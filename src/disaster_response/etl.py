"""ETL: load the raw disaster CSVs, clean them, store them in SQLite.

Usage (from the repository root, after ``pip install -e .``)::

    disaster-etl data/disaster_messages.csv data/disaster_categories.csv \\
        data/DisasterResponse.db

or::

    python -m disaster_response.etl data/disaster_messages.csv \\
        data/disaster_categories.csv data/DisasterResponse.db
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd
from sqlalchemy import create_engine

#: Columns that are message metadata, not classification targets.
METADATA_COLUMNS = ["id", "message", "original", "genre"]


def load_data(messages_filepath: str, categories_filepath: str) -> pd.DataFrame:
    """Read the messages and categories CSVs and join them on ``id``."""
    messages = pd.read_csv(messages_filepath, dtype=str)
    categories = pd.read_csv(categories_filepath, dtype=str)
    return messages.merge(categories, how="inner", on="id")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Expand the ``categories`` column into one binary column per category.

    Each ``categories`` value looks like
    ``"related-1;request-0;offer-0;..."``. The column is split on ``;``,
    the ``name-value`` pairs become 36 binary columns, and any value
    other than 0/1 (the dataset contains a few ``related-2`` rows) is
    clipped to 1. Duplicate rows are dropped.
    """
    categories = df["categories"].str.split(";", expand=True)
    # Column names come from the first row, e.g. "related-1" -> "related".
    categories.columns = categories.iloc[0].str.rsplit("-", n=1).str[0]
    for column in categories.columns:
        categories[column] = (
            categories[column]
            .str.rsplit("-", n=1)
            .str[1]
            .astype(int)
            .clip(lower=0, upper=1)
        )

    df = df.drop(columns=["categories"])
    df = pd.concat([df, categories], axis=1)
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def save_data(df: pd.DataFrame, database_filepath: str) -> str:
    """Write ``df`` to a SQLite table named after the database file.

    Returns the table name used.
    """
    table_name = os.path.splitext(os.path.basename(database_filepath))[0]
    engine = create_engine(f"sqlite:///{database_filepath}")
    try:
        df.to_sql(table_name, engine, index=False, if_exists="replace")
    finally:
        engine.dispose()
    return table_name


def run_etl(
    messages_filepath: str, categories_filepath: str, database_filepath: str
) -> tuple[pd.DataFrame, str]:
    """Run the full ETL and return ``(cleaned_df, table_name)``."""
    print(
        f"Loading data...\n    MESSAGES: {messages_filepath}\n"
        f"    CATEGORIES: {categories_filepath}"
    )
    df = load_data(messages_filepath, categories_filepath)

    print("Cleaning data...")
    df = clean_data(df)
    print(f"    {len(df):,} rows, {len(df.columns)} columns")

    print(f"Saving data...\n    DATABASE: {database_filepath}")
    table_name = save_data(df, database_filepath)
    print(f"Cleaned data saved to table '{table_name}'!")
    return df, table_name


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Clean disaster message data and store it in SQLite."
    )
    parser.add_argument("messages", help="Path to disaster_messages.csv")
    parser.add_argument("categories", help="Path to disaster_categories.csv")
    parser.add_argument("database", help="SQLite file to write, e.g. data/DisasterResponse.db")
    args = parser.parse_args(argv)
    run_etl(args.messages, args.categories, args.database)
    return 0


if __name__ == "__main__":
    sys.exit(main())

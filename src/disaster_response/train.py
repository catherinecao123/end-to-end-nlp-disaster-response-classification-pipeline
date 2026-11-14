"""Training CLI: fit the classifier on the ETL database and save it.

Usage (from the repository root, after ``pip install -e .``)::

    # Quick smoke run: small grid, 3,000-row subsample
    disaster-train data/DisasterResponse.db models/classifier.pkl --limit 3000

    # Full run: the full hyperparameter grid on all rows
    disaster-train data/DisasterResponse.db models/classifier.pkl --grid full
"""

from __future__ import annotations

import argparse
import sys

from sklearn.model_selection import train_test_split

from disaster_response.model import (
    build_model,
    evaluate_model,
    load_data,
    save_model,
)
from disaster_response.text import ensure_nltk_resources


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train the disaster response classifier."
    )
    parser.add_argument("database", help="SQLite file produced by the ETL step")
    parser.add_argument("model", help="Where to save the trained model (.pkl)")
    parser.add_argument(
        "--grid",
        choices=["small", "full"],
        default="small",
        help="Hyperparameter grid: 'small' (default, one configuration) "
        "or 'full' (2x2 search, much slower).",
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=2,
        help="Cross-validation folds for grid search (default: 2).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Train on only the first N rows (smoke runs).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data held out for evaluation (default: 0.2).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for the train/test split (default: 42).",
    )
    args = parser.parse_args(argv)

    ensure_nltk_resources()

    print(f"Loading data...\n    DATABASE: {args.database}")
    X, y, category_names = load_data(args.database, limit=args.limit)
    print(f"    {len(X):,} messages, {len(category_names)} categories")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    print(f"Building model (grid={args.grid}, cv={args.cv})...")
    model = build_model(grid=args.grid, cv=args.cv)

    print("Training model...")
    model.fit(X_train, y_train)

    print("Evaluating model...")
    evaluate_model(model, X_test, y_test, category_names)

    print(f"Saving model...\n    MODEL: {args.model}")
    save_model(model, args.model)

    print("Trained model saved!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

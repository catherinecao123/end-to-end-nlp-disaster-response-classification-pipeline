"""Model: multi-output text classifier for disaster messages.

Pipeline: ``CountVectorizer`` (custom tokenizer) + ``TfidfTransformer``,
combined via ``FeatureUnion`` with a message-length feature, feeding a
``MultiOutputClassifier`` of random forests — one binary classifier per
category. Hyperparameters are tuned with ``GridSearchCV``.
"""

from __future__ import annotations

import os

import joblib
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sqlalchemy import create_engine

from disaster_response.etl import METADATA_COLUMNS
from disaster_response.text import TextLengthExtractor, tokenize

#: Small grid (default): a single configuration, for smoke runs and CI.
#: ``n_jobs=-1`` parallelizes the CV folds.
SMALL_GRID = {
    "clf__estimator__n_estimators": [50],
    "clf__estimator__min_samples_split": [2],
}

#: Full grid: the original 2x2 search. Slower; use ``--grid full``.
FULL_GRID = {
    "clf__estimator__n_estimators": [50, 100],
    "clf__estimator__min_samples_split": [2, 3],
}

PARAM_GRIDS = {"small": SMALL_GRID, "full": FULL_GRID}


def load_data(
    database_filepath: str, limit: int | None = None
) -> tuple[pd.Series, pd.DataFrame, list[str]]:
    """Load features, targets and category names from the SQLite database.

    Parameters
    ----------
    database_filepath:
        Path to the ``.db`` file produced by the ETL step.
    limit:
        If set, only the first ``limit`` rows are loaded (smoke runs).

    Returns
    -------
    X:
        Message texts.
    y:
        Binary target frame, one column per category.
    category_names:
        The category column names, in ``y`` order.
    """
    table_name = os.path.splitext(os.path.basename(database_filepath))[0]
    engine = create_engine(f"sqlite:///{database_filepath}")
    try:
        query = f'SELECT * FROM "{table_name}"'
        if limit:
            query += f" LIMIT {int(limit)}"
        df = pd.read_sql_query(query, con=engine)
    finally:
        engine.dispose()

    X = df["message"]
    y = df.drop(columns=METADATA_COLUMNS)
    return X, y, list(y.columns)


def build_model(grid: str = "small", cv: int = 2) -> GridSearchCV:
    """Build the GridSearchCV-wrapped classification pipeline.

    Parameters
    ----------
    grid:
        ``"small"`` (default) or ``"full"``; see :data:`PARAM_GRIDS`.
    cv:
        Number of cross-validation folds.
    """
    if grid not in PARAM_GRIDS:
        raise ValueError(f"Unknown grid {grid!r}; choose from {sorted(PARAM_GRIDS)}")

    pipeline = Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        (
                            "text_pipeline",
                            Pipeline(
                                [
                                    (
                                        "vect",
                                        CountVectorizer(
                                            tokenizer=tokenize, token_pattern=None
                                        ),
                                    ),
                                    ("tfidf", TfidfTransformer()),
                                ]
                            ),
                        ),
                        ("text_length", TextLengthExtractor()),
                    ]
                ),
            ),
            (
                "clf",
                MultiOutputClassifier(
                    RandomForestClassifier(n_jobs=-1, random_state=42)
                ),
            ),
        ]
    )
    return GridSearchCV(
        pipeline,
        param_grid=PARAM_GRIDS[grid],
        scoring="recall_micro",
        cv=cv,
        n_jobs=-1,
    )


def evaluate_model(
    model: BaseEstimator,
    X_test: pd.Series,
    y_test: pd.DataFrame,
    category_names: list[str],
) -> dict:
    """Print a classification report and return metrics.

    Returns ``{"report": str, "accuracy": float, "best_params": dict}``.
    Accuracy is the mean per-label correctness across all categories.
    """
    y_pred = model.predict(X_test)
    report = classification_report(
        y_test.values, y_pred, target_names=category_names, zero_division=0
    )
    print(report)
    accuracy = float((y_pred == y_test.values).mean())
    print(f"Mean per-label accuracy: {accuracy:.4f}")
    best_params = getattr(model, "best_params_", {})
    if best_params:
        print(f"Best params: {best_params}")
    return {"report": report, "accuracy": accuracy, "best_params": best_params}


def save_model(model: BaseEstimator, model_filepath: str) -> None:
    """Persist a trained model with joblib (creates parent dirs)."""
    parent = os.path.dirname(os.path.abspath(model_filepath))
    os.makedirs(parent, exist_ok=True)
    joblib.dump(model, model_filepath)


def load_model(model_filepath: str) -> BaseEstimator:
    """Load a model persisted with :func:`save_model`."""
    return joblib.load(model_filepath)

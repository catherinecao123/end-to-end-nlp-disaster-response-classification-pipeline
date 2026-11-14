"""Flask web app: classify a message and visualize the dataset.

Usage (from the repository root, after ``pip install -e .`` and after
running the ETL and training steps)::

    disaster-serve --database data/DisasterResponse.db \\
        --model models/classifier.pkl

then open http://127.0.0.1:3001/.

The app is built with a factory (:func:`create_app`) so tests can pass
in a small database and model instead of the real artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd
import plotly.graph_objects as go
from flask import Flask, render_template, request
from sqlalchemy import create_engine

from disaster_response.etl import METADATA_COLUMNS
from disaster_response.model import load_model

_TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")


def _load_dataframe(database_filepath: str) -> pd.DataFrame:
    table_name = os.path.splitext(os.path.basename(database_filepath))[0]
    engine = create_engine(f"sqlite:///{database_filepath}")
    try:
        return pd.read_sql_table(table_name, con=engine)
    finally:
        engine.dispose()


def _build_graphs(df: pd.DataFrame) -> list[dict]:
    """Build the two Plotly figures shown on the index page.

    Returns JSON-ready figure dicts (``{"data": ..., "layout": ...}``).
    """
    category_columns = [c for c in df.columns if c not in METADATA_COLUMNS]

    genre_counts = df.groupby("genre").count()["message"]
    fig_genres = go.Figure(
        data=[go.Bar(x=list(genre_counts.index), y=genre_counts.tolist())],
        layout={
            "title": "Distribution of Message Genres",
            "yaxis": {"title": "# of messages"},
            "xaxis": {"title": "Genre"},
        },
    )

    # Stacked bars: per-category counts, one trace per genre present.
    ordered_categories = (
        df[category_columns].sum().sort_values(ascending=False).index.tolist()
    )
    traces = [
        go.Bar(
            x=ordered_categories,
            y=df[df["genre"] == genre][ordered_categories].sum().tolist(),
            name=str(genre).title(),
        )
        for genre in sorted(df["genre"].dropna().unique())
    ]
    fig_categories = go.Figure(
        data=traces,
        layout={
            "title": "Message Counts per Category, by Genre",
            "yaxis": {"title": "# of messages"},
            "xaxis": {"title": "Category", "tickangle": -40},
            "barmode": "stack",
        },
    )
    return [json.loads(fig.to_json()) for fig in (fig_genres, fig_categories)]


def create_app(database_filepath: str, model_filepath: str) -> Flask:
    """Create the Flask app bound to a database and a trained model."""
    df = _load_dataframe(database_filepath)
    model = load_model(model_filepath)
    category_columns = [c for c in df.columns if c not in METADATA_COLUMNS]

    app = Flask(__name__, template_folder=_TEMPLATE_DIR)

    @app.route("/")
    @app.route("/index")
    def index():
        graphs = _build_graphs(df)
        ids = [f"graph-{i}" for i, _ in enumerate(graphs)]
        graph_json = json.dumps(graphs)
        return render_template("master.html", ids=ids, graphJSON=graph_json)

    @app.route("/go")
    def go():
        query = request.args.get("query", "")
        classification_labels = model.predict([query])[0]
        classification_results = dict(zip(category_columns, classification_labels))
        return render_template(
            "go.html",
            query=query,
            classification_result=classification_results,
        )

    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Serve the disaster response classifier web app."
    )
    parser.add_argument(
        "--database",
        default="data/DisasterResponse.db",
        help="SQLite file produced by the ETL step.",
    )
    parser.add_argument(
        "--model",
        default="models/classifier.pkl",
        help="Trained model file produced by the training step.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=3001)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)

    app = create_app(args.database, args.model)
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    sys.exit(main())

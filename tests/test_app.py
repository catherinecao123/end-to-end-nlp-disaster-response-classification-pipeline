"""Smoke tests for the Flask app, using a tiny database and model."""

import pytest

from disaster_response.app import create_app
from disaster_response.etl import save_data
from disaster_response.model import build_model, save_model
from tests.conftest import make_tiny_dataset


@pytest.fixture(scope="module")
def app_client(tmp_path_factory):
    """Train a tiny model and build a Flask test client for it."""
    tmp_path = tmp_path_factory.mktemp("app")
    df = make_tiny_dataset(n_per_class=6)
    db_path = tmp_path / "Tiny.db"
    save_data(df, str(db_path))

    X = df["message"]
    y = df[["related", "water", "food", "shelter"]]
    model = build_model(grid="small", cv=2)
    model.fit(X, y)
    model_path = tmp_path / "model.pkl"
    save_model(model, str(model_path))

    app = create_app(str(db_path), str(model_path))
    app.config["TESTING"] = True
    return app.test_client()


def test_index_returns_200(app_client):
    response = app_client.get("/")
    assert response.status_code == 200
    assert b"Disaster Response" in response.data
    assert b"graph-0" in response.data
    assert b"graph-1" in response.data


def test_go_classifies_query(app_client):
    response = app_client.get("/go", query_string={"query": "we need water"})
    assert response.status_code == 200
    assert b"we need water" in response.data
    # Category names render title-cased.
    assert b"Water" in response.data
    assert b"Shelter" in response.data


def test_go_empty_query_returns_200(app_client):
    response = app_client.get("/go")
    assert response.status_code == 200

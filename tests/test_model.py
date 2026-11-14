"""Tests for the model: build, train/predict smoke, evaluate, persist."""

import numpy as np
import pytest

from disaster_response.model import (
    build_model,
    evaluate_model,
    load_model,
    save_model,
)
from tests.conftest import make_tiny_dataset


@pytest.fixture(scope="module")
def tiny_xy():
    df = make_tiny_dataset(n_per_class=6)
    X = df["message"]
    y = df[["related", "water", "food", "shelter"]]
    return X, y, list(y.columns)


def test_build_model_rejects_unknown_grid():
    with pytest.raises(ValueError, match="Unknown grid"):
        build_model(grid="huge")


def test_train_predict_smoke(tiny_xy):
    X, y, _ = tiny_xy
    model = build_model(grid="small", cv=2)
    model.fit(X, y)
    predictions = model.predict(X)
    assert predictions.shape == (len(X), y.shape[1])
    assert set(np.unique(predictions)) <= {0, 1}
    assert model.best_params_ == {
        "clf__estimator__min_samples_split": 2,
        "clf__estimator__n_estimators": 50,
    }


def test_evaluate_model_returns_metrics(tiny_xy, capsys):
    X, y, category_names = tiny_xy
    model = build_model(grid="small", cv=2)
    model.fit(X, y)
    metrics = evaluate_model(model, X, y, category_names)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert "water" in metrics["report"]
    assert metrics["best_params"] == model.best_params_
    assert "accuracy" in capsys.readouterr().out.lower()


def test_save_load_model_roundtrip(tiny_xy, tmp_path):
    X, y, _ = tiny_xy
    model = build_model(grid="small", cv=2)
    model.fit(X, y)
    model_path = tmp_path / "nested" / "model.pkl"
    save_model(model, str(model_path))
    assert model_path.exists()
    reloaded = load_model(str(model_path))
    before = model.predict(X[:5])
    after = reloaded.predict(X[:5])
    assert (before == after).all()

"""Smoke tests for the locked models.py interface, on synthetic data."""

import numpy as np
import pandas as pd
import pytest

from src.models import FEATURES, FEATURES_FULL, fit_evaluate, predict_grid


def _fake_sites(n=200, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({f: rng.normal(size=n).astype("float32") for f in FEATURES})
    df["sample_id"] = [f"S{i:04d}" for i in range(n)]
    df["rock_class"] = rng.integers(0, 4, size=n).astype("int16")
    df["fold"] = rng.integers(0, 5, size=n).astype("int8")
    df["as_log"] = df["mag_tmi"] * 0.8 + df["k_pct"] * 0.3 + rng.normal(scale=0.2, size=n)
    return df


@pytest.mark.parametrize("model", ["rf", "ridge", "dummy"])
def test_fit_evaluate_regression(model):
    out = fit_evaluate(_fake_sites(), "as_log", FEATURES_FULL, model, task="reg")
    assert list(out["scores"].columns) == ["fold", "rmse", "r2"]
    assert len(out["scores"]) == 5
    assert list(out["oof"].columns) == ["sample_id", "y_true", "y_pred"]
    assert list(out["importance"].columns) == ["feature", "importance_mean", "importance_std"]


@pytest.mark.parametrize("model", ["rf", "mlr", "dummy"])
def test_fit_evaluate_classification(model):
    out = fit_evaluate(_fake_sites(), "rock_class", FEATURES, model, task="clf")
    assert list(out["scores"].columns) == ["fold", "accuracy", "f1_macro"]


def test_rf_beats_dummy_on_signal():
    df = _fake_sites()
    rf = fit_evaluate(df, "as_log", FEATURES, "rf")["scores"]["r2"].mean()
    dummy = fit_evaluate(df, "as_log", FEATURES, "dummy")["scores"]["r2"].mean()
    assert rf > dummy


def test_predict_grid_shape():
    train = _fake_sites()
    grid = _fake_sites(50, seed=1).drop(columns=["sample_id", "as_log", "fold"])
    grid["cell_id"] = np.arange(50, dtype="int64")
    out = predict_grid(train, "as_log", FEATURES_FULL, grid)
    assert list(out.columns) == ["cell_id", "y_pred"]
    assert len(out) == 50


def test_unknown_model_rejected():
    with pytest.raises(ValueError):
        fit_evaluate(_fake_sites(), "as_log", FEATURES, "mlr", task="reg")

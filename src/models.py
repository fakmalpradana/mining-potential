"""Shared model functions for Option D (Tellus mineral potential mapping).

The interface is locked on 24 Nov 2026 (task #8). Change it only with the
whole team's agreement, because Gavin and Lanbai call these functions.

All supervised models are validated with spatial block cross-validation:
every row must carry a ``fold`` column (0-4) from ``sample_folds.parquet``
or ``grid_folds.parquet``. Never use a random split.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
GRID_TEMPLATE = Path(__file__).resolve().parents[1] / "data" / "processed" / "grid_250m.tif"

# The 13 model features, exactly as named in features_soil_sites.parquet.
FEATURES = [
    "mag_tmi",
    "mag_hgrad",
    "mag_vd1",
    "k_pct",
    "eu_ppm",
    "eth_ppm",
    "k_eth",
    "eu_eth",
    "cond",
    "elev_m",
    "slope_deg",
    "grav_bouguer",
    "dist_fault_m",
]

# Core model features: FEATURES plus one-hot rock_class.
FEATURES_FULL = FEATURES + ["rock_class"]

CATEGORICAL = ["rock_class"]


def _design_matrix(df: pd.DataFrame, features: list[str], columns: list[str] | None = None) -> pd.DataFrame:
    """Select features and one-hot encode categorical ones (rock_class)."""
    X = df[features].copy()
    cats = [c for c in CATEGORICAL if c in features]
    if cats:
        X = pd.get_dummies(X, columns=cats, prefix=cats, dtype=float)
    if columns is not None:
        X = X.reindex(columns=columns, fill_value=0.0)
    return X.astype(float)


def _make_model(model: str, task: str):
    if task == "reg":
        options = {
            "rf": lambda: RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=RANDOM_STATE),
            "ridge": lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13))),
            "dummy": lambda: DummyRegressor(strategy="mean"),
        }
    elif task == "clf":
        options = {
            "rf": lambda: RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=RANDOM_STATE),
            "mlr": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
            "dummy": lambda: DummyClassifier(strategy="most_frequent"),
        }
    else:
        raise ValueError("task must be 'reg' or 'clf'")
    if model not in options:
        raise ValueError(f"model '{model}' is not available for task '{task}': choose from {sorted(options)}")
    return options[model]()


def fit_evaluate(df: pd.DataFrame, target: str, features: list[str], model: str, task: str = "reg") -> dict:
    """Train and score a model with spatial block cross-validation.

    Parameters
    ----------
    df : DataFrame with the feature columns, the target column, ``fold`` and,
        for soil sites, ``sample_id``.
    target : name of the target column, for example ``as_log`` or ``rock_class``.
    features : ``FEATURES`` or ``FEATURES_FULL`` (or a subset for ablations).
    model : ``rf``, ``ridge``, ``mlr`` or ``dummy``.
    task : ``reg`` (regression) or ``clf`` (classification).

    Returns
    -------
    dict with
      ``scores``: DataFrame ``fold, rmse, r2`` (reg) or ``fold, accuracy, f1_macro`` (clf)
      ``oof``: DataFrame ``sample_id, y_true, y_pred`` (out-of-fold predictions)
      ``importance``: DataFrame ``feature, importance_mean, importance_std`` (permutation)
    """
    missing = [c for c in [*features, target, "fold"] if c not in df.columns]
    if missing:
        raise KeyError(f"missing columns: {missing}")
    data = df.dropna(subset=[target]).reset_index(drop=True)
    ids = data["sample_id"] if "sample_id" in data.columns else data.get("cell_id", pd.Series(data.index))
    X_all = _design_matrix(data, features)
    y_all = data[target]

    rows, oof, importances = [], [], []
    for fold in sorted(data["fold"].unique()):
        test = data["fold"] == fold
        est = _make_model(model, task)
        est.fit(X_all[~test], y_all[~test])
        pred = est.predict(X_all[test])
        y_true = y_all[test]
        if task == "reg":
            rows.append({"fold": int(fold),
                         "rmse": float(np.sqrt(mean_squared_error(y_true, pred))),
                         "r2": float(r2_score(y_true, pred))})
        else:
            rows.append({"fold": int(fold),
                         "accuracy": float(accuracy_score(y_true, pred)),
                         "f1_macro": float(f1_score(y_true, pred, average="macro"))})
        oof.append(pd.DataFrame({"sample_id": ids[test].values, "y_true": y_true.values, "y_pred": pred}))
        pi = permutation_importance(est, X_all[test], y_true, n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1)
        importances.append(pd.Series(pi.importances_mean, index=X_all.columns))

    imp = pd.concat(importances, axis=1)
    importance = pd.DataFrame({"feature": imp.index,
                               "importance_mean": imp.mean(axis=1).values,
                               "importance_std": imp.std(axis=1).fillna(0).values})
    importance = importance.sort_values("importance_mean", ascending=False).reset_index(drop=True)
    return {"scores": pd.DataFrame(rows), "oof": pd.concat(oof, ignore_index=True), "importance": importance}


def predict_grid(df_train: pd.DataFrame, target: str, features: list[str], grid_df: pd.DataFrame,
                 model: str = "rf") -> pd.DataFrame:
    """Train on all of ``df_train`` and predict every cell of ``features_grid250.parquet``.

    Returns a DataFrame ``cell_id, y_pred``.
    """
    data = df_train.dropna(subset=[target])
    X = _design_matrix(data, features)
    est = _make_model(model, "reg")
    est.fit(X, data[target])
    X_grid = _design_matrix(grid_df, features, columns=list(X.columns))
    return pd.DataFrame({"cell_id": grid_df["cell_id"].values, "y_pred": est.predict(X_grid)})


def to_raster(pred_df: pd.DataFrame, value_col: str, out_path: str | Path, template: str | Path = GRID_TEMPLATE) -> Path:
    """Write predictions to a GeoTIFF on the ``grid_250m.tif`` grid.

    ``cell_id`` = row * width + col of the template. Output follows the
    general rules: float32, 1 band, nodata -9999, same transform and CRS.
    """
    import rasterio

    with rasterio.open(template) as src:
        profile = src.profile.copy()
        height, width = src.height, src.width
    arr = np.full(height * width, -9999.0, dtype="float32")
    arr[pred_df["cell_id"].to_numpy(dtype="int64")] = pred_df[value_col].to_numpy(dtype="float32")
    profile.update(dtype="float32", count=1, nodata=-9999, compress="deflate")
    out_path = Path(out_path)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr.reshape(height, width), 1)
    return out_path

"""Tests for the data-contract checker, using small synthetic files."""

import numpy as np
import pandas as pd

from src.check_contract import FEATURES, check_file


def _soil_sites(n=10):
    df = pd.DataFrame({"sample_id": [f"S{i}" for i in range(n)],
                       "x": np.arange(n, dtype="float64"), "y": np.arange(n, dtype="float64")})
    for f in FEATURES:
        df[f] = np.ones(n, dtype="float32")
    df["rock_class"] = np.zeros(n, dtype="int16")
    df["rock_name"] = "basalt"
    return df


def test_valid_feature_table_passes(tmp_path):
    p = tmp_path / "features_soil_sites.parquet"
    _soil_sites().to_parquet(p)
    status, errors = check_file(p)
    assert status == "pass", errors


def test_wrong_dtype_and_missing_column_fail(tmp_path):
    df = _soil_sites().drop(columns=["cond"])
    df["k_pct"] = df["k_pct"].astype("float64")
    p = tmp_path / "features_soil_sites.parquet"
    df.to_parquet(p)
    status, errors = check_file(p)
    assert status == "fail"
    assert any("cond" in e for e in errors)
    assert any("k_pct" in e for e in errors)


def test_nan_in_feature_fails(tmp_path):
    df = _soil_sites()
    df.loc[0, "mag_tmi"] = np.nan
    p = tmp_path / "features_soil_sites.parquet"
    df.to_parquet(p)
    status, errors = check_file(p)
    assert status == "fail"


def test_geochem_requires_triplets(tmp_path):
    n = 5
    df = pd.DataFrame({"sample_id": [f"S{i}" for i in range(n)],
                       "x": np.zeros(n), "y": np.zeros(n),
                       "in_nw": True, "in_se": False, "in_ne": False, "in_mafic": False})
    for el in ("as", "au"):
        df[f"{el}_raw"] = np.ones(n, dtype="float32")
        df[f"{el}_log"] = np.zeros(n, dtype="float32")
        df[f"{el}_bdl"] = False
    p = tmp_path / "geochem_clean.parquet"
    df.to_parquet(p)
    assert check_file(p)[0] == "pass"
    df.drop(columns=["au_bdl"]).to_parquet(p)
    assert check_file(p)[0] == "fail"


def test_unknown_file_is_skipped(tmp_path):
    p = tmp_path / "notes.csv"
    p.write_text("a\n1\n")
    assert check_file(p)[0] == "skip"

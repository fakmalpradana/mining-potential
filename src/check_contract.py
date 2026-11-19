"""Check a hand-over file against the Option D data contract (Section 9 of the Work Plan).

Usage
-----
    python src/check_contract.py data/processed/features_soil_sites.parquet
    python src/check_contract.py --all data/processed      # every contract file found

Exit code 0 = every checked file passes, 1 = at least one failure.
A file whose name is not in the contract is reported and skipped.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GRID_TEMPLATE = ROOT / "data" / "processed" / "grid_250m.tif"

FEATURES = ["mag_tmi", "mag_hgrad", "mag_vd1", "k_pct", "eu_ppm", "eth_ppm", "k_eth", "eu_eth",
            "cond", "elev_m", "slope_deg", "grav_bouguer", "dist_fault_m"]
F32 = {f: "float32" for f in FEATURES}

# Tables: exact columns (order-free) and dtypes. "string" accepts pandas object/string.
TABLES = {
    "features_soil_sites.parquet": {
        "columns": {"sample_id": "string", "x": "float64", "y": "float64", **F32,
                    "rock_class": "int16", "rock_name": "string"},
        "key": "sample_id", "no_nan": FEATURES + ["rock_class"],
    },
    "features_grid250.parquet": {
        "columns": {"cell_id": "int64", "row": "int32", "col": "int32", "x": "float64", "y": "float64", **F32,
                    "rock_class": "int16", "rock_name": "string"},
        "key": "cell_id", "no_nan": FEATURES + ["rock_class"],
    },
    "sample_folds.parquet": {"columns": {"sample_id": "string", "block_id": "int32", "fold": "int8"}, "key": "sample_id"},
    "grid_folds.parquet": {"columns": {"cell_id": "int64", "block_id": "int32", "fold": "int8"}, "key": "cell_id"},
    "kmeans_grid.parquet": {"columns": {"cell_id": "int64", "cluster": "int16"}, "key": "cell_id"},
    "oof_core.parquet": {"columns": {"sample_id": "string", "target": "string", "y_true": "float", "y_pred": "float"}},
    "rock_classes.csv": {"columns": {"rock_class": "int", "rock_name": "string", "original_units": "string", "n_points": "int"}},
    "scores_rock.csv": {"columns": {"model": "string", "fold": "int", "accuracy": "float", "f1_macro": "float"}, "rows": 15},
    "scores_core.csv": {
        "columns": {"target": "string", "model": "string", "fold": "int", "rmse": "float", "r2": "float"},
        "rows": 30,
    },
    "importance_core.csv": {
        "columns": {"target": "string", "feature": "string", "importance_mean": "float", "importance_std": "float"},
    },
    "deposit_test.csv": {"columns": {"metric": "string", "value": "float"}},
    "ablation_gravity.csv": {"columns": {"target": "string", "with_gravity": "bool", "r2_mean": "float", "r2_std": "float"}},
    "element_scores.csv": {"columns": {"element": "string", "region": "string", "with_rock": "bool",
                                       "r2_mean": "float", "r2_std": "float", "rmse_mean": "float",
                                       "n_samples": "int", "pct_bdl": "float"}},
    "kmeans_ari.csv": {"columns": {"k": "int", "ari": "float"}},
}

# Rasters: dtype, band count, nodata. All must match the grid_250m.tif template.
RASTERS = {
    "grid_250m.tif": {"dtype": "uint8", "count": 1, "nodata": None},
    "gravity_bouguer.tif": {"dtype": "float32", "count": 1, "nodata": -9999},
    "dist_to_fault.tif": {"dtype": "float32", "count": 1, "nodata": -9999},
    "potential_as.tif": {"dtype": "float32", "count": 2, "nodata": -9999},
    "rock_pred.tif": {"dtype": "int16", "count": 1, "nodata": -1},
}

# Vectors: required attribute columns.
VECTORS = {
    "deposit_holdout.gpkg": ["name", "radius_m"],
    "spatial_blocks.gpkg": ["block_id", "fold"],
    "regions.gpkg": ["region"],
}

GEOCHEM_FIXED = {"sample_id": "string", "x": "float64", "y": "float64",
                 "in_nw": "bool", "in_se": "bool", "in_ne": "bool", "in_mafic": "bool"}
ELEMENTS = ["as", "au", "cu", "pb", "zn", "ni", "cr", "pt", "pd"]


def _dtype_ok(series: pd.Series, expected: str) -> bool:
    kind = series.dtype
    if expected == "string":
        return pd.api.types.is_string_dtype(kind) or pd.api.types.is_object_dtype(kind)
    if expected == "float":
        return pd.api.types.is_float_dtype(kind)
    if expected == "int":
        return pd.api.types.is_integer_dtype(kind)
    if expected == "bool":
        return pd.api.types.is_bool_dtype(kind)
    return str(kind) == expected


def _read_table(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def check_table(path: Path, spec: dict) -> list[str]:
    errors = []
    df = _read_table(path)
    expected = spec["columns"]
    missing = sorted(set(expected) - set(df.columns))
    extra = sorted(set(df.columns) - set(expected))
    if missing:
        errors.append(f"missing columns: {missing}")
    if extra:
        errors.append(f"unexpected columns: {extra}")
    for col, dt in expected.items():
        if col in df.columns and not _dtype_ok(df[col], dt):
            errors.append(f"column '{col}' has dtype {df[col].dtype}, expected {dt}")
    for col in spec.get("no_nan", []):
        if col in df.columns and df[col].isna().any():
            errors.append(f"column '{col}' contains {int(df[col].isna().sum())} NaN values")
    key = spec.get("key")
    if key and key in df.columns and df[key].duplicated().any():
        errors.append(f"key '{key}' has duplicated values")
    if "rows" in spec and len(df) != spec["rows"]:
        errors.append(f"expected {spec['rows']} rows, found {len(df)}")
    return errors


def check_geochem(path: Path) -> list[str]:
    errors = []
    df = pd.read_parquet(path)
    for col, dt in GEOCHEM_FIXED.items():
        if col not in df.columns:
            errors.append(f"missing column '{col}'")
        elif not _dtype_ok(df[col], dt):
            errors.append(f"column '{col}' has dtype {df[col].dtype}, expected {dt}")
    pattern = re.compile(r"^([a-z]{1,2})_(raw|log|bdl)$")
    found = {}
    for col in df.columns:
        if col in GEOCHEM_FIXED:
            continue
        m = pattern.match(col)
        if not m or m.group(1) not in ELEMENTS:
            errors.append(f"unexpected column '{col}' (expected <el>_raw, <el>_log or <el>_bdl)")
            continue
        found.setdefault(m.group(1), set()).add(m.group(2))
        want = "bool" if m.group(2) == "bdl" else "float32"
        if not _dtype_ok(df[col], want):
            errors.append(f"column '{col}' has dtype {df[col].dtype}, expected {want}")
    for el, kinds in found.items():
        if kinds != {"raw", "log", "bdl"}:
            errors.append(f"element '{el}' needs {el}_raw, {el}_log and {el}_bdl")
    for el in ("as", "au"):
        if el not in found:
            errors.append(f"element '{el}' is required")
    if df["sample_id"].duplicated().any() if "sample_id" in df.columns else False:
        errors.append("key 'sample_id' has duplicated values")
    return errors


def check_raster(path: Path, spec: dict) -> list[str]:
    import rasterio

    errors = []
    with rasterio.open(path) as src:
        if src.dtypes[0] != spec["dtype"]:
            errors.append(f"dtype {src.dtypes[0]}, expected {spec['dtype']}")
        if src.count != spec["count"]:
            errors.append(f"{src.count} bands, expected {spec['count']}")
        if spec["nodata"] is not None and src.nodata != spec["nodata"]:
            errors.append(f"nodata {src.nodata}, expected {spec['nodata']}")
        if path.name != "grid_250m.tif":
            if not GRID_TEMPLATE.exists():
                errors.append("grid_250m.tif template not found in data/processed, cannot check alignment")
            else:
                with rasterio.open(GRID_TEMPLATE) as tpl:
                    if (src.width, src.height) != (tpl.width, tpl.height):
                        errors.append(f"shape {src.height}x{src.width}, template is {tpl.height}x{tpl.width}")
                    if not src.transform.almost_equals(tpl.transform):
                        errors.append("transform differs from grid_250m.tif")
                    if src.crs != tpl.crs:
                        errors.append(f"CRS {src.crs}, template is {tpl.crs}")
        elif abs(src.res[0] - 250) > 1e-6 or abs(src.res[1] - 250) > 1e-6:
            errors.append(f"pixel size {src.res}, expected 250 m")
    return errors


def check_vector(path: Path, columns: list[str]) -> list[str]:
    import geopandas as gpd

    errors = []
    gdf = gpd.read_file(path)
    missing = [c for c in columns if c not in gdf.columns]
    if missing:
        errors.append(f"missing columns: {missing}")
    if gdf.crs is None:
        errors.append("no CRS set")
    if GRID_TEMPLATE.exists() and gdf.crs is not None:
        import rasterio

        with rasterio.open(GRID_TEMPLATE) as tpl:
            if gdf.crs != tpl.crs:
                errors.append(f"CRS {gdf.crs}, team CRS (grid_250m.tif) is {tpl.crs}")
    return errors


def check_file(path: Path) -> tuple[str, list[str]]:
    name = path.name
    if not path.exists():
        return "fail", [f"file not found: {path}"]
    if name == "geochem_clean.parquet":
        return _result(check_geochem(path))
    if name in TABLES:
        return _result(check_table(path, TABLES[name]))
    if name in RASTERS:
        return _result(check_raster(path, RASTERS[name]))
    if name in VECTORS:
        return _result(check_vector(path, VECTORS[name]))
    return "skip", [f"'{name}' is not a contract file"]


def _result(errors: list[str]) -> tuple[str, list[str]]:
    return ("fail" if errors else "pass"), errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", help="files to check, or a folder with --all")
    parser.add_argument("--all", action="store_true", help="check every contract file inside the given folder(s)")
    args = parser.parse_args(argv)

    files: list[Path] = []
    for p in map(Path, args.paths):
        if args.all and p.is_dir():
            names = set(TABLES) | set(RASTERS) | set(VECTORS) | {"geochem_clean.parquet"}
            files += sorted(f for f in p.rglob("*") if f.name in names)
        else:
            files.append(p)
    if not files:
        print("No contract files found, nothing to check.")
        return 0

    failed = False
    for f in files:
        status, messages = check_file(f)
        print(f"[{status.upper()}] {f}")
        for m in messages:
            print(f"    - {m}")
        failed |= status == "fail"
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

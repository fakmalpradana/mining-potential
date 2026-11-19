# data/processed

Hand-over files between members go here, named exactly as in Section 9 of the Work Plan.
`.tif`, `.gpkg` and `.parquet` files are git-ignored: upload them to the shared Drive and copy them into this folder locally.
Small `.csv` results (scores, element tables) are committed, and CI checks them against the contract on every push.

Before handing over any file, run:

```bash
python src/check_contract.py data/processed/<file>
```

| File | Owner | Due | Recipient |
| --- | --- | --- | --- |
| `grid_250m.tif`, `gravity_bouguer.tif`, `dist_to_fault.tif` | Hyatma | 21 Nov | Gavin, Akmal |
| `deposit_holdout.gpkg` | Hyatma | 21 Nov | Akmal |
| `geochem_clean.parquet`, `regions.gpkg` | Lanbai | 22 Nov | Akmal |
| `spatial_blocks.gpkg`, `sample_folds.parquet`, `grid_folds.parquet` | Akmal | 22 Nov | Gavin, Lanbai |
| `features_soil_sites.parquet`, `features_grid250.parquet`, `rock_classes.csv` | Gavin | 24 Nov | Lanbai, Akmal |
| `kmeans_grid.parquet`, `kmeans_ari.csv` | Lanbai | 28 Nov | Notebook |
| `scores_rock.csv`, `rock_pred.tif` | Gavin | 1 Dec | Akmal |
| `scores_core.csv`, `oof_core.parquet`, `importance_core.csv` | Akmal | 1 Dec | Report |
| `potential_as.tif`, `deposit_test.csv`, `ablation_gravity.csv` | Akmal | 2 Dec | Hyatma |
| `element_scores.csv` | Lanbai | 12 Dec | Report |

# Mining potential: ML mineral potential mapping from Tellus data

**EARTH 5014 Machine Learning Applications to Earth System Problems (2026–27), Option D**
Team: Hyatma, Gavin, Lanbai, Akmal (coordinator)

We map gold and other mineral potential in Northern Ireland from the Tellus airborne geophysics and soil geochemistry surveys, using a streamlined machine-learning workflow. Presentation on **4 December 2026**; report (1,200 ± 20% words + notebook appendix) due **17 December 2026**.

## Project summary

A Random Forest learns how airborne measurements (magnetics, radiometrics, conductivity, gravity) relate to arsenic and gold concentrations in soil samples. The trained model then predicts concentrations for every 250 m grid cell, and the highest predictions form the mineral potential map. The Curraghinalt–Cavanacaw gold deposit area is hidden during training to test whether the model still flags it. The same pipeline is repeated for Cu, Zn, Ni, Cr, Pt and Pd to see which metals airborne data can predict, and whether that is only because they follow rock type.

This is an ML applications project: geology is the context, ML is the subject. Three messages run through the work:

1. Random validation on spatial data gives over-optimistic scores, so every model uses **spatial block cross-validation** (10–20 km blocks, 5 folds).
2. A complex model is only worth using if it **beats a simple baseline**, and the margin is reported.
3. A high score can come from a **shortcut** (for example Ni following the basalt map), so ablations and interpretation separate signal from shortcut.

### Research questions

1. **RQ1** How accurately can airborne geophysics classify the mapped bedrock units?
2. **RQ2** How well can arsenic and gold be predicted at locations the model has never seen?
3. **RQ3** Does a model trained without the deposit area still rank that area as high potential?
4. **RQ4** Which elements can be predicted from the air, and is that only because they follow rock type?

### Data

| Dataset | Use |
| --- | --- |
| Tellus airborne geophysics (magnetics, radiometrics, EM) | Model features |
| Tellus rural soil geochemistry (aqua regia, fire assay) | Targets: As, Au and other elements |
| GSNI 1:250k bedrock geology and faults | Rock-type labels, distance to faults |
| EIGEN-6C4 global gravity model (ICGEM) | Regional gravity feature |

Download instructions: [`data/raw/README.md`](data/raw/README.md).

### Models

| Model | Role |
| --- | --- |
| Random Forest regressor | Core model: predicts element concentrations, produces the potential map |
| Random Forest classifier | Classifies rock type to test whether the features carry geological signal |
| Ridge regression | Linear baseline for regression |
| Multinomial logistic regression | Linear baseline for rock classification |
| Dummy (mean / majority class) | Lower bound for every score |
| PCA | Summarises correlated geophysical features |
| k-means | Unsupervised groups, compared with the geology map (ARI) |

### Final outputs

- Mineral potential map on a 250 m grid (`potential_as.tif`, Figure 4)
- Hidden-deposit test, model scores vs baselines, per-element predictability (Figures 1–5)
- 10-minute presentation and 1,200-word report with this repo's `main.ipynb` as the appendix

## Team and responsibilities

| Member | Role | Main outputs |
| --- | --- | --- |
| Hyatma | Data provider and science lead | Raw data, GIS layers, sense check, Figure 4, introduction and discussion |
| Gavin | Geophysics processing | Feature tables, rock classification, Figures 1–2 |
| Lanbai | Geochemistry and exploratory analysis | Clean geochemistry, EDA, element loop, Figure 5 |
| Akmal | Model and validation owner, coordinator | Spatial blocks, `models.py`, core model, potential map, Figure 3, `main.ipynb` |

The full plan, task board and deadlines are in [`docs/Option_D_Work_Plan.docx`](docs/Option_D_Work_Plan.docx). Each member has a personal guide in `docs/Guide_<Name>.pdf` with a to-do checklist, step-by-step instructions and an output checklist. The overall flowchart is [`docs/Option_D_Flowchart.png`](docs/Option_D_Flowchart.png).

## Repository structure

```
mining-potential/
├── data/
│   ├── raw/                raw downloads (on Drive, git-ignored)
│   └── processed/          hand-over files named as in the data contract
├── src/
│   ├── models.py           shared model functions (interface locked 24 Nov)
│   └── check_contract.py   checks a hand-over file against the data contract
├── notebooks/              one working notebook per member
├── main.ipynb              main notebook, sections 01–08, assembled by Akmal
├── figures/                Figures 1–5 (PNG, 300 dpi) + source CSVs
├── report/                 report sections, one file per member
├── docs/                   work plan, personal guides, flowchart
└── tests/                  unit tests run by CI
```

## Getting started

```bash
git clone https://github.com/fakmalpradana/mining-potential.git
cd mining-potential
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q                          # should pass before you start
git checkout -b <your-name>        # hyatma, gavin, lanbai or akmal
```

Then copy the data you need from the shared Drive into `data/raw/` and `data/processed/`, open `notebooks/<your-name>.ipynb`, and follow your guide.

## Working together

1. **One branch per member.** Work on `hyatma`, `gavin`, `lanbai` or `akmal`, never directly on `main`.
2. **Open a pull request to `main`** when a step is done. Akmal reviews and merges after CI passes.
3. **Large data stays on the Drive.** `.tif`, `.gpkg` and `.parquet` are git-ignored; small `.csv` results are committed.
4. **Check every hand-over file** before you report it:
   ```bash
   python src/check_contract.py data/processed/<file>
   ```
5. **Clear notebook outputs** before committing (Jupyter: Edit → Clear All Outputs).
6. **Report in the team chat** with `[DONE #<task number>] <file> → <recipient>` and update the task board. Blocked? Tell Akmal at least one day before the deadline.

### Using the shared model functions

```python
from src.models import FEATURES, FEATURES_FULL, fit_evaluate, predict_grid, to_raster

out = fit_evaluate(df, target="as_log", features=FEATURES_FULL, model="rf", task="reg")
out["scores"]       # fold, rmse, r2
out["oof"]          # sample_id, y_true, y_pred
out["importance"]   # feature, importance_mean, importance_std

grid_pred = predict_grid(df, "as_log", FEATURES_FULL, grid_df, model="rf")   # cell_id, y_pred
to_raster(grid_pred, "y_pred", "data/processed/potential_as_band1.tif")
```

`df` must contain the feature columns, the target and a `fold` column (from `sample_folds.parquet` or `grid_folds.parquet`). Models: `rf`, `ridge`, `dummy` for `task="reg"`; `rf`, `mlr`, `dummy` for `task="clf"`.

## Continuous integration

Every push and pull request runs [`.github/workflows/ci.yml`](.github/workflows/ci.yml):

| Step | What it checks |
| --- | --- |
| Lint | `ruff check src tests` |
| Notebooks | every `.ipynb` is valid and has no committed outputs |
| Unit tests | the locked `models.py` interface and the contract checker still behave as agreed |
| Data contract | every committed hand-over file in `data/processed/` matches Section 9 |

A red CI means the change would break someone else's work: fix it before merging.

## Data licence

Tellus and GSNI geology data © Geological Survey of Northern Ireland, used under the UK Open Government Licence. Gravity computed from EIGEN-6C4 (Förste et al., 2014) via the ICGEM Calculation Service, GFZ Potsdam.

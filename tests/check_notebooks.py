"""Fail if any notebook is invalid JSON or still contains cell outputs.

Clear outputs before you commit (Jupyter: Edit -> Clear All Outputs), so
diffs stay small and merges do not conflict. The final PDF export is made
from a fresh run of main.ipynb, not from committed outputs.
"""

import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
bad = []
for path in sorted(ROOT.rglob("*.ipynb")):
    if ".ipynb_checkpoints" in path.parts:
        continue
    try:
        nb = nbformat.read(path, as_version=4)
        nbformat.validate(nb)
    except Exception as exc:  # noqa: BLE001
        bad.append(f"{path.relative_to(ROOT)}: invalid notebook ({exc})")
        continue
    if any(c.get("outputs") for c in nb.cells if c.cell_type == "code"):
        bad.append(f"{path.relative_to(ROOT)}: has cell outputs, clear them before committing")

for line in bad:
    print(line)
sys.exit(1 if bad else 0)

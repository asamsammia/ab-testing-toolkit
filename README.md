# A/B Testing Toolkit

Reusable experiment analysis toolkit with **power checks**, **CUPED adjustment**, and clean **statsmodels** readouts.

## Why
Speed up the loop from raw logs → clean metrics → decision. Standardizes guardrails and reduces manual errors.

## Tech
- Python 3.10+, pandas, numpy, statsmodels, scipy, jupyter
- Optional: matplotlib, plotly for visuals

## Data
Works with generic event logs:
```
user_id, timestamp, variant, metric_1, metric_2, ...
```
Adapt the loader in `src/data_loader.py` to your schema.

## How to Run
```bash
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt  # or pip install pandas numpy statsmodels scipy
jupyter lab  # open notebooks in /notebooks
```
- Start with `notebooks/01_power_and_design.ipynb` (power calc, CUPED setup)
- Then `notebooks/02_analysis.ipynb` (effect sizes, CIs, sanity checks)
- Export a one-page HTML/PDF summary from `notebooks/03_report.ipynb`

## Key Results (example placeholders)
- Cut time from cleaned data → decision by **~X%** on sample workloads.
- Reduced variance with CUPED by **~Y%** vs. vanilla diff-in-means.

## Repo Map
- `src/exp/` experiment helpers (power, CUPED, validate)
- `src/metrics/` KPI definitions and tests
- `notebooks/` example workflows
- `tests/` unit tests

## Notes
- Replace X/Y with real numbers once you run on your dataset.
- Add a small synthetic CSV to `data/sample/` if allowed.

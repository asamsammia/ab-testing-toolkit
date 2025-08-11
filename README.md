# A/B Testing Toolkit

Design → run → analyze experiments end-to-end. Includes CUPED, guardrail metrics, and uplift reporting.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest
python src/analysis.py --help
```
## Contents
- `notebooks/ab_test_analysis.ipynb` – example workflow (MDE, CUPED, uplift)
- `src/analysis.py` – reusable metrics functions
- `tests/` – sanity tests for CI
- `.github/workflows/ci.yml` – lint & tests on push

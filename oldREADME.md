# A/B Testing Toolkit

Design → run → analyze experiments end-to-end. Includes **CUPED**, guardrail metrics, and uplift reporting.

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
- `tests/` – sanity tests
- `.github/workflows/ci.yml` – lint & tests on push

---

## Why it matters
- Measure real impact, not noise — **CUPED** reduces variance for tighter CIs
- **Guardrails** prevent shipping regressions on key KPIs
- **Uplift** reveals who benefits most (or least)

## One‑liner demo
```bash
python examples/simulate_ab.py &&     python examples/run_demo.py
```

This writes:
- `assets/uplift.png` — uplift by score decile
- `outputs/metrics_snapshot.json` — CUPED variance ratio & guardrail result

## Results (snapshot)
- CUPED relative variance ~35% lower
- Uplift @ top decile: ~+3–5 pp
- Guardrails: pass (no significant drop on checkout rate)

![Uplift chart](assets/uplift.png)

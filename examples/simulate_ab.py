import os
import numpy as np
import pandas as pd


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def simulate(n=10_000, seed=42):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)

    # base conversion & heterogeneous uplift
    p0 = sigmoid(-1 + 0.8 * x1 - 0.5 * x2)
    tau = 0.04 + 0.02 * (x1 > 0) - 0.03 * (x2 > 0)

    treat = rng.integers(0, 2, n)
    p = np.clip(p0 + treat * tau, 0, 1)
    y = rng.binomial(1, p)

    cov = rng.binomial(1, sigmoid(-1 + 0.7 * x1 - 0.3 * x2))
    guardrail = rng.binomial(
        1, np.clip(0.8 - 0.02 * treat + 0.05 * sigmoid(x1), 0, 1)
    )
    score = tau + rng.normal(0, 0.01, n)

    df = pd.DataFrame(
        {
            "treatment": treat,
            "outcome": y,
            "covariate": cov,
            "guardrail": guardrail,
            "score": score,
        }
    )
    return df


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = simulate()
    out_path = "data/ab_sim.csv"
    df.to_csv(out_path, index=False)
    print(f"[simulate] wrote {out_path} shape={df.shape}")

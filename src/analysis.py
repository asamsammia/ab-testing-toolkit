import argparse
import numpy as np
from scipy import stats


def lift(control, treatment):
    c = np.mean(control)
    t = np.mean(treatment)
    return (t - c) / c if c != 0 else np.nan


def mde(alpha=0.05, power=0.8, p=0.1, n_per_group=1000):
    """Approximate MDE for proportion metric."""
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    se = (2 * p * (1 - p) / n_per_group) ** 0.5
    return (z_alpha + z_beta) * se / p


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A/B metrics quick helpers")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run a quick demo computation"
    )
    args = parser.parse_args()
    if args.demo:
        rng = np.random.default_rng(0)
        control = rng.normal(100, 10, 1000)
        treatment = rng.normal(103, 10, 1000)
        print("Lift:", round(lift(control, treatment), 4))
        print("MDE:", round(mde(), 4))

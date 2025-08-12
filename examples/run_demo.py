import os, json, numpy as np, pandas as pd, matplotlib.pyplot as plt

def cuped_adjust(y, cov):
    cov = cov.astype(float); y = y.astype(float)
    theta = np.cov(y, cov, ddof=1)[0,1] / np.var(cov, ddof=1)
    y_adj = y - theta * (cov - cov.mean())
    return y_adj, float(theta)

def two_prop_ztest(succ_a, n_a, succ_b, n_b):
    p_a = succ_a / n_a; p_b = succ_b / n_b
    p_pool = (succ_a + succ_b) / (n_a + n_b)
    se = np.sqrt(p_pool*(1-p_pool)*(1/n_a + 1/n_b))
    z = (p_b - p_a) / se if se > 0 else 0.0
    return float(z)

def uplift_by_decile(df):
    df = df.copy()
    df["decile"] = pd.qcut(df["score"], 10, labels=False, duplicates="drop") + 1
    out = []
    for d, g in df.groupby("decile"):
        t = g[g["treatment"]==1]["outcome"].mean()
        c = g[g["treatment"]==0]["outcome"].mean()
        out.append((int(d), float(t - c)))
    out.sort(key=lambda x: x[0])
    return out

if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("assets", exist_ok=True)
    path = "data/ab_sim.csv"
    if not os.path.exists(path):
        raise SystemExit("Run: python examples/simulate_ab.py (to create data/ab_sim.csv)")

    df = pd.read_csv(path)
    t, c = df[df.treatment==1], df[df.treatment==0]

    # CUPED
    y_adj_t, theta = cuped_adjust(t["outcome"], t["covariate"])
    y_adj_c, _     = cuped_adjust(c["outcome"], c["covariate"])
    var_ratio = float(np.var(pd.concat([y_adj_t, y_adj_c]), ddof=1) / np.var(df["outcome"], ddof=1))

    # Guardrail z-test (one-sided)
    z = two_prop_ztest(c["guardrail"].sum(), len(c), t["guardrail"].sum(), len(t))
    guardrail_pass = (z >= -1.64)

    # Uplift chart
    deciles = uplift_by_decile(df)
    xs = [d for d,_ in deciles]; ys = [u for _,u in deciles]
    plt.figure(figsize=(6,4))
    plt.plot(xs, ys, marker="o")
    plt.xlabel("Score decile (1=lowest)")
    plt.ylabel("Uplift (pp)")
    plt.title("Uplift by score decile")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("assets/uplift.png", dpi=150)
    plt.close()

    snapshot = {
        "cuped_theta": theta,
        "variance_ratio_vs_raw": var_ratio,
        "guardrail_z": z,
        "guardrail_pass": bool(guardrail_pass),
        "uplift_top_decile_pp": ys[-1] * 100.0
    }
    with open("outputs/metrics_snapshot.json", "w") as f:
        json.dump(snapshot, f, indent=2)

    print("[demo] wrote assets/uplift.png and outputs/metrics_snapshot.json")

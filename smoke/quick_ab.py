
import csv, math, statistics, sys

def read_data(path):
    control, treatment = [], []
    with open(path, newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            m = float(row['metric'])
            if row['variant'].lower() == 'control':
                control.append(m)
            else:
                treatment.append(m)
    return control, treatment

def mean_ci(vals, z=1.96):
    n = len(vals)
    mu = statistics.mean(vals)
    sd = statistics.pstdev(vals) if n>1 else 0.0
    se = sd / math.sqrt(n) if n>0 else 0.0
    return mu, (mu - z*se, mu + z*se), sd

def two_sample_z(control, treatment):
    n1, n2 = len(control), len(treatment)
    mu1, mu2 = statistics.mean(control), statistics.mean(treatment)
    v1 = statistics.pvariance(control) if n1>1 else 0.0
    v2 = statistics.pvariance(treatment) if n2>1 else 0.0
    se = math.sqrt(v1/n1 + v2/n2)
    z = (mu2 - mu1) / se if se>0 else 0.0
    # Normal approx p-value (two-sided)
    # Using error function for CDF
    def phi(x):
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))
    p = 2 * (1 - phi(abs(z)))
    return (mu1, mu2, mu2-mu1, z, p)

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv)>1 else "sample_ab.csv"
    control, treatment = read_data(path)
    mu1, ci1, sd1 = mean_ci(control)
    mu2, ci2, sd2 = mean_ci(treatment)
    mu_c, mu_t, diff, z, p = two_sample_z(control, treatment)
    print(f"Control n={len(control)} mean={mu1:.3f} CI95=({ci1[0]:.3f},{ci1[1]:.3f})")
    print(f"Treatment n={len(treatment)} mean={mu2:.3f} CI95=({ci2[0]:.3f},{ci2[1]:.3f})")
    print(f"Diff (T-C)={diff:.3f}  z={z:.3f}  approx p={p:.5f}")
    if p<0.05:
        print("Result: Statistically significant at 5% level (normal approx).")
    else:
        print("Result: Not statistically significant at 5% level (normal approx).")

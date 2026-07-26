"""When is a federated subgroup calibration estimate actually trustworthy?

The federated aggregation in fedcal.py reproduces the pooled ECE exactly. That
says nothing about whether the pooled ECE is itself a reliable number when a
demographic subgroup is small.

This script measures that. It builds a model that is perfectly calibrated by
construction -- outcomes are drawn with exactly the predicted probability, so
the true ECE is zero -- and then asks what ECE we actually measure at different
subgroup sizes. Any value above zero is pure small-sample bias.

Run:  python reliability_experiment.py
"""

import json
import math

import numpy as np

import fedcal


TRIALS = 400
SIZES = [10, 25, 50, 100, 250, 500, 1000, 5000, 20000]
BIN_COUNTS = [5, 10, 20, 50]


def measure_ece(rng, n, n_bins):
    """One trial: perfectly calibrated model, measured through the federated path."""
    p = rng.uniform(0.0, 1.0, n)
    y = (rng.random(n) < p).astype(float)
    groups = np.full(n, "G")
    report = fedcal.compute_client_report("site", p, y, groups, n_bins=n_bins)
    merged = fedcal.merge_reports([report])
    return fedcal.subgroup_metrics(merged, min_n_warn=1)["G"]["ece"]


def bias_by_size(rng, n_bins=10):
    rows = []
    for n in SIZES:
        values = np.array([measure_ece(rng, n, n_bins) for _ in range(TRIALS)])
        rows.append(
            {
                "n": n,
                "mean_ece": float(values.mean()),
                "p05": float(np.percentile(values, 5)),
                "p95": float(np.percentile(values, 95)),
            }
        )
    return rows


def bias_by_bins(rng, n=200):
    rows = []
    for n_bins in BIN_COUNTS:
        values = np.array([measure_ece(rng, n, n_bins) for _ in range(TRIALS)])
        rows.append({"n_bins": n_bins, "mean_ece": float(values.mean())})
    return rows


def fit_power_law(rows):
    """Bias should fall off like n**(-0.5) if it is ordinary sampling noise."""
    n = np.array([r["n"] for r in rows], dtype=float)
    ece = np.array([r["mean_ece"] for r in rows], dtype=float)
    slope, intercept = np.polyfit(np.log(n), np.log(ece), 1)
    return float(slope), float(np.exp(intercept))


def predicted_floor(n, n_bins):
    """Closed-form estimate of the ECE a perfectly calibrated model will show.

    Within a bin holding n_b patients, the observed outcome rate differs from
    the predicted rate by sampling noise with standard deviation
    sqrt(p(1-p)/n_b). The expected absolute value of a mean-zero normal
    deviation is sqrt(2/pi) times its standard deviation. Weighting each bin by
    n_b/n and assuming predictions spread evenly over [0, 1], where the average
    of sqrt(p(1-p)) is pi/8, this collapses to

        bias  ~  sqrt(2/pi) * (pi/8) * sqrt(n_bins / n)

    The sqrt(n_bins / n) scaling is general. The leading constant assumes the
    uniform prediction spread used here; a cohort whose predictions cluster near
    zero, as in a rare-event setting, has a smaller constant.
    """
    return math.sqrt(2.0 / math.pi) * (math.pi / 8.0) * math.sqrt(n_bins / n)


def main():
    rng = np.random.default_rng(20260726)

    size_rows = bias_by_size(rng)
    bin_rows = bias_by_bins(rng)
    slope, coefficient = fit_power_law(size_rows)

    print("A perfectly calibrated model has a true ECE of exactly 0.")
    print(f"Measured ECE, 10 bins, {TRIALS} trials per row:\n")
    print(f"{'subgroup n':>11} | {'mean ECE':>9} | {'5th-95th pct':>18} | {'predicted':>9}")
    print("-" * 57)
    for row in size_rows:
        span = f"{row['p05']:.4f} - {row['p95']:.4f}"
        print(
            f"{row['n']:>11} | {row['mean_ece']:>9.4f} | {span:>18} "
            f"| {predicted_floor(row['n'], 10):>9.4f}"
        )

    print(f"\nFitted scaling: bias ~ {coefficient:.3f} * n^({slope:.3f})")
    print("An exponent near -0.5 means this is ordinary sampling noise,")
    print("not a defect in the model or in the federated aggregation.\n")

    print(f"Effect of bin count at n=200, {TRIALS} trials per row:\n")
    print(f"{'bins':>6} | {'mean ECE':>9} | {'predicted':>9}")
    print("-" * 31)
    for row in bin_rows:
        predicted = predicted_floor(200, row["n_bins"])
        print(f"{row['n_bins']:>6} | {row['mean_ece']:>9.4f} | {predicted:>9.4f}")
    print("\nMore bins means fewer patients per bin, so more bins inflates the")
    print("bias. Bin count is not a neutral choice.\n")

    floor = next(r for r in size_rows if r["n"] == 250)
    print("Reading for the thyroid case: a subgroup of roughly 250 patients still")
    print(f"shows a spurious ECE of about {floor['mean_ece']:.3f} even when the model")
    print("is flawless. Any measured subgroup ECE below that is indistinguishable")
    print("from perfect calibration.")

    with open("reliability_results.json", "w") as handle:
        json.dump(
            {
                "trials": TRIALS,
                "by_size": size_rows,
                "by_bins": bin_rows,
                "power_law": {"exponent": slope, "coefficient": coefficient},
            },
            handle,
            indent=2,
        )


if __name__ == "__main__":
    main()

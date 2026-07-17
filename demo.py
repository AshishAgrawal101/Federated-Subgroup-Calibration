"""Run the prototype on four synthetic hospitals."""

import numpy as np

from fedcal import (
    compute_client_report,
    merge_reports,
    pooled_subgroup_metrics,
    simulate_hospital,
    subgroup_metrics,
)


rng = np.random.default_rng(42)

sites = {
    "large_urban": simulate_hospital(rng, 2000, 0.30, {"F": 0.7, "M": 0.3}),
    "midsize": simulate_hospital(
        rng,
        500,
        0.15,
        {"F": 0.5, "M": 0.5},
        miscalibration={"M": (0.5, 1.8)},
    ),
    "small_regional": simulate_hospital(
        rng, 120, 0.45, {"F": 0.85, "M": 0.15}
    ),
    "tiny_clinic": simulate_hospital(rng, 40, 0.25, {"F": 0.4, "M": 0.6}),
}

reports = [
    compute_client_report(site_id, *site_data)
    for site_id, site_data in sites.items()
]
federated = subgroup_metrics(merge_reports(reports))

pooled = pooled_subgroup_metrics(
    np.concatenate([site[0] for site in sites.values()]),
    np.concatenate([site[1] for site in sites.values()]),
    np.concatenate([site[2] for site in sites.values()]),
)

print("Local cohort sizes:")
for site_id, (probability, _, group) in sites.items():
    counts = {str(label): int((group == label).sum()) for label in np.unique(group)}
    print(f"  {site_id:15s} n={len(probability):5d}  {counts}")

print()
print(
    f"{'group':<8} {'n':>7} {'fed ECE':>12} {'pooled ECE':>12} "
    f"{'fed Brier':>12} {'pooled Brier':>13}"
)
for group in sorted(federated):
    print(
        f"{group:<8} {federated[group]['n']:>7} "
        f"{federated[group]['ece']:>12.6f} {pooled[group]['ece']:>12.6f} "
        f"{federated[group]['brier']:>12.6f} {pooled[group]['brier']:>13.6f}"
    )

max_difference = max(
    max(
        abs(federated[group]["ece"] - pooled[group]["ece"]),
        abs(federated[group]["brier"] - pooled[group]["brier"]),
    )
    for group in federated
)
print(f"\nmax |federated - pooled| = {max_difference:.2e}")

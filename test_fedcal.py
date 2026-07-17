"""Tests for the first calibration prototype."""

import unittest

import numpy as np

from fedcal import (
    compute_client_report,
    merge_reports,
    pooled_subgroup_metrics,
    simulate_hospital,
    subgroup_metrics,
)


def make_sites(seed=0):
    rng = np.random.default_rng(seed)
    return [
        simulate_hospital(rng, 2000, 0.30, {"F": 0.7, "M": 0.3}),
        simulate_hospital(
            rng,
            500,
            0.15,
            {"F": 0.5, "M": 0.5},
            miscalibration={"M": (0.5, 1.8)},
        ),
        simulate_hospital(rng, 120, 0.45, {"F": 0.85, "M": 0.15}),
        simulate_hospital(rng, 40, 0.25, {"F": 0.4, "M": 0.6}),
    ]


class FedCalTests(unittest.TestCase):
    def test_federated_matches_pooled(self):
        sites = make_sites()
        reports = [
            compute_client_report(f"hospital_{i}", *site)
            for i, site in enumerate(sites)
        ]
        federated = subgroup_metrics(merge_reports(reports))
        pooled = pooled_subgroup_metrics(
            np.concatenate([site[0] for site in sites]),
            np.concatenate([site[1] for site in sites]),
            np.concatenate([site[2] for site in sites]),
        )

        for group in pooled:
            self.assertEqual(federated[group]["n"], pooled[group]["n"])
            self.assertAlmostEqual(federated[group]["ece"], pooled[group]["ece"], 12)
            self.assertAlmostEqual(
                federated[group]["brier"], pooled[group]["brier"], 12
            )

    def test_small_group_warning(self):
        rng = np.random.default_rng(1)
        site = simulate_hospital(rng, 60, group_probs={"F": 0.9, "M": 0.1})
        result = subgroup_metrics(
            merge_reports([compute_client_report("site", *site)]),
            min_n_warn=50,
        )
        self.assertIsNotNone(result["M"]["warning"])

    def test_report_contains_aggregates_not_arrays(self):
        rng = np.random.default_rng(2)
        site = simulate_hospital(rng, 10_000)
        report = compute_client_report("large", *site)
        occupied_cells = sum(len(bins) for bins in report.cells.values())
        self.assertLessEqual(occupied_cells, 20)
        self.assertFalse(hasattr(report, "y_prob"))
        self.assertFalse(hasattr(report, "y_true"))

    def test_mismatched_bins_are_rejected(self):
        rng = np.random.default_rng(3)
        site = simulate_hospital(rng, 100)
        reports = [
            compute_client_report("a", *site, n_bins=10),
            compute_client_report("b", *site, n_bins=15),
        ]
        with self.assertRaises(ValueError):
            merge_reports(reports)

    def test_nan_probability_is_rejected(self):
        with self.assertRaises(ValueError):
            compute_client_report("a", [np.nan], [1], ["F"])

    def test_nonbinary_outcome_is_rejected(self):
        with self.assertRaises(ValueError):
            compute_client_report("a", [0.5], [2], ["F"])

    def test_empty_cohort_is_rejected(self):
        with self.assertRaises(ValueError):
            compute_client_report("a", [], [], [])

    def test_invalid_bin_count_is_rejected(self):
        with self.assertRaises(ValueError):
            compute_client_report("a", [0.5], [1], ["F"], n_bins=0)

    def test_duplicate_client_id_is_rejected(self):
        first = compute_client_report("same", [0.2], [0], ["F"])
        second = compute_client_report("same", [0.8], [1], ["M"])
        with self.assertRaises(ValueError):
            merge_reports([first, second])

    def test_invalid_simulation_settings_are_rejected(self):
        rng = np.random.default_rng(4)
        with self.assertRaises(ValueError):
            simulate_hospital(rng, 100, group_probs={"F": 0.7, "M": 0.2})
        with self.assertRaises(ValueError):
            simulate_hospital(rng, 100, outcome_rate=1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""Small prototype for federated subgroup calibration evaluation."""

from dataclasses import dataclass, field
from numbers import Integral

import numpy as np


DEFAULT_N_BINS = 10


@dataclass(frozen=True)
class CellStats:
    n: int = 0
    sum_p: float = 0.0
    sum_y: float = 0.0
    sum_sq_err: float = 0.0

    def add(self, other):
        return CellStats(
            self.n + other.n,
            self.sum_p + other.sum_p,
            self.sum_y + other.sum_y,
            self.sum_sq_err + other.sum_sq_err,
        )


@dataclass
class ClientReport:
    client_id: str
    n_bins: int = DEFAULT_N_BINS
    cells: dict = field(default_factory=dict)


def _check_n_bins(n_bins):
    if isinstance(n_bins, bool) or not isinstance(n_bins, Integral) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer")
    return int(n_bins)


def _check_client_id(client_id):
    if not isinstance(client_id, str) or not client_id.strip():
        raise ValueError("client_id must be a non-empty string")
    return client_id.strip()


def _check_arrays(y_prob, y_true, groups):
    try:
        y_prob = np.asarray(y_prob, dtype=float)
        y_true = np.asarray(y_true, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("predictions and outcomes must be numeric") from error
    groups = np.asarray(groups, dtype=object)

    if y_prob.ndim != 1 or y_true.ndim != 1 or groups.ndim != 1:
        raise ValueError("predictions, outcomes, and groups must be one-dimensional")
    if not (len(y_prob) == len(y_true) == len(groups)):
        raise ValueError("predictions, outcomes, and groups must have equal length")
    if len(y_prob) == 0:
        raise ValueError("evaluation arrays cannot be empty")
    if not np.all(np.isfinite(y_prob)):
        raise ValueError("predictions must contain only finite values")
    if np.any((y_prob < 0) | (y_prob > 1)):
        raise ValueError("predictions must lie in [0, 1]")
    if not np.all(np.isfinite(y_true)) or not np.all(np.isin(y_true, [0, 1])):
        raise ValueError("outcomes must contain only 0 and 1")

    clean_groups = []
    for value in groups.tolist():
        if value is None:
            raise ValueError("group labels cannot be missing")
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            raise ValueError("group labels cannot be missing")
        label = str(value).strip()
        if not label:
            raise ValueError("group labels cannot be empty")
        clean_groups.append(label)

    return y_prob, y_true, np.asarray(clean_groups, dtype=str)


def compute_client_report(
    client_id,
    y_prob,
    y_true,
    groups,
    n_bins=DEFAULT_N_BINS,
):
    """Convert one hospital's predictions into group/bin totals."""

    client_id = _check_client_id(client_id)
    n_bins = _check_n_bins(n_bins)
    y_prob, y_true, groups = _check_arrays(y_prob, y_true, groups)
    bin_index = np.minimum((y_prob * n_bins).astype(int), n_bins - 1)

    report = ClientReport(client_id=client_id, n_bins=n_bins)
    for group in np.unique(groups):
        group_mask = groups == group
        report.cells[group] = {}
        for bin_number in np.unique(bin_index[group_mask]):
            mask = group_mask & (bin_index == bin_number)
            p = y_prob[mask]
            y = y_true[mask]
            report.cells[group][int(bin_number)] = CellStats(
                n=int(mask.sum()),
                sum_p=float(p.sum()),
                sum_y=float(y.sum()),
                sum_sq_err=float(((p - y) ** 2).sum()),
            )
    return report


def _check_cell(group, bin_number, cell, n_bins):
    if not isinstance(group, str) or not group:
        raise ValueError("report groups must be non-empty strings")
    if isinstance(bin_number, bool) or not isinstance(bin_number, Integral):
        raise ValueError("bin indices must be integers")
    if not 0 <= int(bin_number) < n_bins:
        raise ValueError("report contains a bin outside its configured range")
    if not isinstance(cell, CellStats):
        raise ValueError("report cells must contain CellStats")
    if isinstance(cell.n, bool) or not isinstance(cell.n, Integral) or cell.n < 1:
        raise ValueError("each occupied cell must have a positive count")

    sums = np.asarray([cell.sum_p, cell.sum_y, cell.sum_sq_err], dtype=float)
    if not np.all(np.isfinite(sums)) or np.any(sums < 0):
        raise ValueError("report sums must be finite and non-negative")
    tolerance = 1e-12 * max(1, int(cell.n))
    if np.any(sums > cell.n + tolerance):
        raise ValueError("a report sum cannot exceed its cell count")


def merge_reports(reports):
    """Validate and add reports from several hospitals."""

    if not reports:
        raise ValueError("no reports to merge")
    if any(not isinstance(report, ClientReport) for report in reports):
        raise ValueError("all reports must be ClientReport objects")

    n_bins = _check_n_bins(reports[0].n_bins)
    seen_clients = set()
    merged = {}

    for report in reports:
        client_id = _check_client_id(report.client_id)
        if client_id in seen_clients:
            raise ValueError(f"duplicate client_id: {client_id}")
        seen_clients.add(client_id)
        if _check_n_bins(report.n_bins) != n_bins:
            raise ValueError("all clients must use the same bins")
        if not isinstance(report.cells, dict) or not report.cells:
            raise ValueError("each report must contain at least one group")

        for group, bins in report.cells.items():
            if not isinstance(bins, dict) or not bins:
                raise ValueError("each report group must contain an occupied bin")
            merged.setdefault(group, {})
            for bin_number, cell in bins.items():
                _check_cell(group, bin_number, cell, n_bins)
                old = merged[group].get(int(bin_number), CellStats())
                merged[group][int(bin_number)] = old.add(cell)
    return merged


def subgroup_metrics(merged, min_n_warn=50):
    """Calculate ECE and Brier score from merged cells."""

    if not isinstance(min_n_warn, Integral) or isinstance(min_n_warn, bool):
        raise ValueError("min_n_warn must be a positive integer")
    if min_n_warn < 1:
        raise ValueError("min_n_warn must be a positive integer")
    if not isinstance(merged, dict) or not merged:
        raise ValueError("merged statistics cannot be empty")

    result = {}
    for group, bins in merged.items():
        group_n = sum(cell.n for cell in bins.values())
        ece = sum(abs(cell.sum_p - cell.sum_y) for cell in bins.values()) / group_n
        brier = sum(cell.sum_sq_err for cell in bins.values()) / group_n
        result[group] = {
            "n": group_n,
            "ece": float(ece),
            "brier": float(brier),
            "warning": (
                f"only {group_n} patients; estimate may be unreliable"
                if group_n < min_n_warn
                else None
            ),
        }
    return result


def pooled_subgroup_metrics(
    y_prob,
    y_true,
    groups,
    n_bins=DEFAULT_N_BINS,
):
    """Reference calculation used to test the federated result."""

    n_bins = _check_n_bins(n_bins)
    y_prob, y_true, groups = _check_arrays(y_prob, y_true, groups)
    bin_index = np.minimum((y_prob * n_bins).astype(int), n_bins - 1)

    result = {}
    for group in np.unique(groups):
        group_mask = groups == group
        p = y_prob[group_mask]
        y = y_true[group_mask]
        bins = bin_index[group_mask]
        group_n = int(group_mask.sum())
        ece = 0.0
        for bin_number in np.unique(bins):
            mask = bins == bin_number
            ece += (mask.sum() / group_n) * abs(p[mask].mean() - y[mask].mean())
        result[group] = {
            "n": group_n,
            "ece": float(ece),
            "brier": float(((p - y) ** 2).mean()),
        }
    return result


def simulate_hospital(
    rng,
    n_patients,
    outcome_rate=0.3,
    group_probs=None,
    miscalibration=None,
):
    """Generate a synthetic hospital cohort for testing."""

    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be a numpy random generator")
    if isinstance(n_patients, bool) or not isinstance(n_patients, Integral):
        raise ValueError("n_patients must be a positive integer")
    if n_patients < 1:
        raise ValueError("n_patients must be a positive integer")
    if not np.isfinite(outcome_rate) or not 0 < outcome_rate < 1:
        raise ValueError("outcome_rate must be between 0 and 1")

    group_probs = {"F": 0.5, "M": 0.5} if group_probs is None else group_probs
    if not isinstance(group_probs, dict) or not group_probs:
        raise ValueError("group_probs must be a non-empty dictionary")

    names = [str(name).strip() for name in group_probs]
    probabilities = np.asarray(list(group_probs.values()), dtype=float)
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("group names must be non-empty and unique")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0):
        raise ValueError("group probabilities must be finite and non-negative")
    if not np.isclose(probabilities.sum(), 1.0, atol=1e-12):
        raise ValueError("group probabilities must sum to 1")

    miscalibration = {} if miscalibration is None else miscalibration
    if not isinstance(miscalibration, dict):
        raise ValueError("miscalibration must be a dictionary")
    if set(miscalibration).difference(names):
        raise ValueError("miscalibration contains an unknown group")
    for parameters in miscalibration.values():
        if not isinstance(parameters, (tuple, list)) or len(parameters) != 2:
            raise ValueError("miscalibration values must be (intercept, slope)")
        if not np.all(np.isfinite(np.asarray(parameters, dtype=float))):
            raise ValueError("miscalibration parameters must be finite")

    groups = rng.choice(names, size=int(n_patients), p=probabilities)
    base_logit = np.log(outcome_rate / (1 - outcome_rate))
    true_logit = base_logit + rng.normal(0, 1.2, size=int(n_patients))
    true_probability = 1 / (1 + np.exp(-true_logit))
    y_true = (rng.random(int(n_patients)) < true_probability).astype(float)

    y_prob = np.empty(int(n_patients), dtype=float)
    for group in names:
        intercept, slope = miscalibration.get(group, (0.0, 1.0))
        mask = groups == group
        y_prob[mask] = 1 / (1 + np.exp(-(intercept + slope * true_logit[mask])))

    return y_prob, y_true, groups

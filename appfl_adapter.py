"""APPFL payload helpers."""

from fedcal import CellStats, ClientReport, merge_reports, subgroup_metrics


ACTION_NAME = "subgroup_calibration"
SCHEMA_VERSION = 1

_REPORT_KEYS = {"schema_version", "client_id", "n_bins", "cells"}
_CELL_KEYS = {"n", "sum_p", "sum_y", "sum_sq_err"}


def make_action_payload(report):
    """Build a serializable report."""

    merge_reports([report])
    return {
        "schema_version": SCHEMA_VERSION,
        "client_id": report.client_id,
        "n_bins": report.n_bins,
        "cells": {
            str(group): {
                str(bin_number): {
                    "n": cell.n,
                    "sum_p": cell.sum_p,
                    "sum_y": cell.sum_y,
                    "sum_sq_err": cell.sum_sq_err,
                }
                for bin_number, cell in bins.items()
            }
            for group, bins in report.cells.items()
        },
    }


def parse_action_payload(payload):
    """Parse and validate a report."""

    if not isinstance(payload, dict) or set(payload) != _REPORT_KEYS:
        raise ValueError("invalid calibration report payload")
    if payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported calibration report schema")
    if not isinstance(payload["cells"], dict):
        raise ValueError("payload cells must be a dictionary")

    cells = {}
    for group, bins in payload["cells"].items():
        if not isinstance(bins, dict):
            raise ValueError("payload group bins must be a dictionary")
        cells[group] = {}
        for bin_number, values in bins.items():
            if not isinstance(values, dict) or set(values) != _CELL_KEYS:
                raise ValueError("invalid calibration cell payload")
            try:
                bin_index = int(bin_number)
            except (TypeError, ValueError) as error:
                raise ValueError("payload bin indices must be integers") from error
            cells[group][bin_index] = CellStats(
                n=values["n"],
                sum_p=values["sum_p"],
                sum_y=values["sum_y"],
                sum_sq_err=values["sum_sq_err"],
            )

    report = ClientReport(
        client_id=payload["client_id"],
        n_bins=payload["n_bins"],
        cells=cells,
    )
    merge_reports([report])
    return report


def aggregate_action_payloads(payloads, min_n_warn=50):
    """Aggregate client reports."""

    reports = [parse_action_payload(payload) for payload in payloads]
    return subgroup_metrics(merge_reports(reports), min_n_warn=min_n_warn)

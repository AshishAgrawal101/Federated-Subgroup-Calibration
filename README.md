# Federated Subgroup Calibration

This project asks whether several hospitals can check a model's calibration
without pooling patient-level predictions in one place.

Each hospital sorts its predictions into the same fixed probability bins and
does this separately for each demographic group. For every occupied bin, a
hospital reports the patient count, the sum of predicted probabilities, the sum
of observed outcomes, and the sum of squared prediction errors. It shares these
totals rather than individual patient rows.

The server adds the totals from all hospitals and calculates expected
calibration error (ECE) and Brier score. ECE compares predicted probabilities
with observed outcome rates, while the Brier score measures the average squared
prediction error.

The ECE calculation in the code is the standard fixed-bin formula written in
additive form. For a bin containing `n_b` patients in a group containing `N`
patients:

```text
(n_b / N) * abs(sum_p / n_b - sum_y / n_b)
    = abs(sum_p - sum_y) / N
```

The `n_b` terms cancel, so the server only needs the bin totals. Summing this
quantity across bins gives the group's fixed-bin ECE.

The metric demo uses four synthetic hospitals with different sample sizes,
outcome rates, demographic distributions, and calibration patterns. Its
federated results match the calculation on the pooled patient-level data up to
floating-point rounding. The difference remains below `1e-16` in the tested
environments.

## How to run

The metric demo, reliability experiment, and tests require NumPy:

```bash
pip install numpy
python demo.py
python reliability_experiment.py
python -m unittest
```

The APPFL integration demo requires Python 3.10:

```bash
pip install appfl==1.10.0
python appfl_grpc_demo.py
```

## APPFL integration

On July 26, 2026, I reproduced APPFL's
[official serial MNIST example](https://appfl.ai/en/stable/tutorials/firstrun.html)
with five simulated clients. I used APPFL v1.10.0 (commit `169ae10`) with Python
3.10. The run completed all ten FedAvg rounds on CPU.

I then connected the calibration calculation to APPFL's gRPC custom-action
interface. In `appfl_grpc_demo.py`, four local APPFL clients each create a
JSON-safe calibration report and send it with
`GRPCClientCommunicator.invoke_custom_action`. The server waits for all four
reports, validates and aggregates them, and returns the same subgroup results
to every client.

This uses APPFL's gRPC client and server communicators. The resulting ECE and
Brier scores match the pooled reference calculation to less than `1e-16`.
The metric code remains separate in `fedcal.py`, while `appfl_adapter.py`
handles the message conversion.

## Small-subgroup reliability experiment

Matching the pooled calculation does not mean every subgroup estimate is
statistically reliable. `reliability_experiment.py` simulates a perfectly
calibrated model, whose true ECE is zero, and repeats the measurement 400 times
at several sample sizes. It also compares 5, 10, 20, and 50 bins.

The experiment shows the positive ECE that can appear from ordinary sampling
noise, especially when a subgroup is small or the number of bins is large. It
writes the full results to `reliability_results.json`; that generated file is
excluded from Git.

## Restrictions

This is still a single-machine simulation. The gRPC server and four clients run
locally without TLS or client authentication. The project demonstrates the
communication and aggregation path, but it is not a production multi-hospital
deployment and does not provide formal privacy protection.

Small totals could reveal information. For example, a demographic group and
probability bin might contain only one or two patients. A real healthcare
network would need additional protections, such as secure aggregation or a
rule that prevents reporting very small groups.

The reliability experiment measures sampling behavior, but it does not provide
formal confidence intervals for real subgroup estimates. Calibration slope and
intercept are also not included. Unlike ECE and Brier score, they require an
iterative model-fitting procedure rather than a single set of aggregated
totals.

## Next steps

The next experiment will vary subgroup prevalence and outcome prevalence as
well as sample size. I also plan to add confidence intervals, run the APPFL
server and clients as separate processes, and evaluate the authentication,
encryption, and small-cell suppression needed for a realistic deployment.

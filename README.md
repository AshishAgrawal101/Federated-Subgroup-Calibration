**Federated Subgroup Calibration**

This simulation was created to answer the question whether it is possible for several hospitals to check a model's calibration without exposing their patient data at a common place?

In the simulation, each hospital sorts its predictions into ten probability ranges which are the same in all hospitals. For example, all the predictions in the range of 0% to 10% will go into one, predictions from 10% to 20% will go into another, and so on.

The hospitals do this by demographic group. For every range that contains patients, they report their numbers of patients, the total predicted probability, the number of outcomes that actually happened, and the total squared prediction error. The only data they share are totals, not individual patient rows.

The server computes the totals from all hospitals and uses them for deriving two measures. The first one is the expected calibration error (ECE) which checks whether the model's confidence is in accordance with the reality. The second is the Brier score which quantifies the magnitude of the model's prediction errors.

I did this in order to see if the same outcome would be achieved by the calculation of the measurements from a combined patient-level dataset. But this is expected, provided that all the hospitals use the same predetermined probability ranges.

In the demo, I created four fake hospitals which exhibit different patterns of behavior. They have different numbers of patients, different outcome rates, different sizes of demographic groups, and different calibration patterns. Depending on the Python and NumPy versions, the final floating-point difference may vary slightly, but it remained below 1e-16 in the tested environments.. As such, the difference is virtually zero. The negligible difference is a result of the way computer programs round off decimal numbers.

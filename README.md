# Federated Subgroup Calibration

This simulation was created to answer the question whether it is possible for several hospitals to check a model's calibration without exposing their patient data at a common place?

In the simulation, each hospital sorts its predictions into ten probability ranges which are the same in all hospitals. For example, all the predictions in the range of 0% to 10% will go into one, predictions from 10% to 20% will go into another, and so on.

The hospitals do this by demographic group. For every range that contains patients, they report their numbers of patients, the total predicted probability, the number of outcomes that actually happened, and the total squared prediction error. The only data they share are totals, not individual patient rows.

The server computes the totals from all hospitals and uses them for deriving two measures. The first one is the expected calibration error (ECE) which checks whether the model's confidence is in accordance with the reality. The second is the Brier score which quantifies the magnitude of the model's prediction errors.

I did this in order to see if the same outcome would be achieved by the calculation of the measurements from a combined patient-level dataset. But this is expected, provided that all the hospitals use the same predetermined probability ranges.

In the demo, I created four fake hospitals which exhibit different patterns of behavior. They have different numbers of patients, different outcome rates, different sizes of demographic groups, and different calibration patterns. With the included random seed, the largest difference between federated and pooled results was 3.47e-17. As such, the difference is virtually zero. The negligible difference is a result of the way computer programs round off decimal numbers.

## Restrictions

This initiative is merely a simulation still. It has not been connected to APPFL and it does not present formal privacy protection as of now.

Small numbers could still give away some information. Suppose there is only one or two patients in a certain demographic group that has been put in a specific probability bin. A real healthcare network should include further protections, such as secure aggregation or a rule that prevents reporting of very small groups.

Matching the pooled ECE also does not assure that the subgroup results are all trustworthy. A result that is based on a small number of patients can still be volatile. The code does show users the warning about the small samples, however, this warning cannot by itself fix the issue. The next phase should be the inclusion of confidence intervals and experiments to examine how sample size impacts the results.

Other than that, the calibration slope and intercept parameter is also on the pipeline. Unlike the ECE and Brier score, they cannot be computed simply by aggregating totals from the hospitals. They require a model-fitting process that runs across multiple patients.

## The following phase

My next task will be to implement the APPFL's serial official example. Following that, I am planning to connect the calibration calculation via APPFL’s custom-action system.

After setting the APPFL version, I will conduct a simulation repeatedly with a changing size of a hospital, subgroup prevalence, and outcome prevalence. As a result, this experiment should indicate the period when federated measurements are trustworthy and the period when small sample sizes are problematic.
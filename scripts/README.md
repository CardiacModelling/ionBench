# Scripts
## Summary
This directory contains python scripts for running ionBench and generating results for the paper. These scripts are described below in the order in which they are most likely to be run, as some scripts build upon results from previous scripts.

## List each script
* summarise what each script does
* what figures it's used for

## Solver Noise
The first script is [solverNoise.py](../scripts/solverNoise.py). This script runs a benchmark problem with a range of solver tolerances to ensure the solver tolerances are appropriate for the problem. The outputs from this script (printed to the command line) that were used to decide the tolerances are saved in the [text](../scripts/text) directory.

We verify the following with this script (where error is the standard deviation of the cost function in a small neighbourhood around a point):
* The median error is below 1e-7 across the whole parameter space,
* The 75th percentile error is below 1e-6 across the whole parameter space,
* The maximum error near the true parameters keeps the cost below the cost threshold.

The script first checks a small number of parameter vectors at a range of solver tolerances and reports which are acceptable. It then asks the user to choose the solver tolerances and runs the same tests with many more parameter vectors.

## Penalty Cost
This script, [penaltyCost](../scripts/penaltyCost.py), runs each of the problems, samples 1000 parameters and reports the maximum cost. This is to verify the minimum penalty cost (1e5) is above the maximum cost of sampled parameters.

## Plot Cost Lines
The script [plotCostLines](../scripts/plotCostLines.py) plots slices of the cost function for each problem. This is used to investigate the cost function and determine how difficult it is likely to be to optimise. The script saves the cost function plots in the [figures/costPlots](../scripts/figures/costPlots) directory.

## Plot Current
This is the first script that generates a plot for the paper. It plots the data and protocol for each problem. The script is [plotCurrent](../scripts/plotCurrent.py), and it saves the plot as [data.png](../scripts/figures/data.png).

## Uncertainty
This is a simple script that calls the profile likelihood function to calculate the data used to plot the profile likelihood plots for each problem. The script is [uncertainty](../scripts/uncertainty.py). It chooses to vary each parameter up and down 10% from the true value and evaluates the profile likelihood at 51 points uniformly spaced between these bounds. It saves the profile likelihood data in the current working directory.

## Plot Profile Likelihood
This plots the previously calculated profile likelihood data. The script is [plotProfileLikelihood](../scripts/plotProfileLikelihood.py). It saves the profile likelihood plots as [profileLikelihood-hh.png](../scripts/figures/profileLikelihood-hh.png), [profileLikelihood-mm.png](../scripts/figures/profileLikelihood-mm.png), [profileLikelihood-ikr.png](../scripts/figures/profileLikelihood-ikr.png), [profileLikelihood-ikur.png](../scripts/figures/profileLikelihood-ikur.png), and [profileLikelihood-ina.png](../scripts/figures/profileLikelihood-ina.png).

The working directory when running this script should be the location of the profile likelihood data.

## Rate Bounds
The script [rateBounds](../scripts/rateBounds.py) generates plots of the rate bounds. It can generate plots from the staircase problems' cost functions to verify they work correctly (`staircase_plots()`), saved under (TODO: reference plots). It also produces a plot of the rate bounds for the paper (`paper_plot()`), saved as [rateBounds.png](../scripts/figures/rateBounds.png).

## Run Optimisers
This script, [runOptimisers](../scripts/runOptimisers.py), runs each unique approach on a given problem. It uses 5 sampled parameters and starting locations. A seed is set to ensure the same starting locations are used. The results are saved in the current working directory.

## Review Optimiser Runs
This script, [reviewOptimiserRuns](../scripts/reviewOptimiserRuns.py), loads the optimisation results and generates *.csv* files for each problem containing summary statistic information of the approaches and optimisation runs. Each approach records the timing information, best cost, and the number of function evaluations for each run. For each approach, we also derive the expected run time (in seconds, cost evals, and grad evals), success rate, tier (tier=1 for success rate>0, tier=2 for success rate=0), and the number of function evaluations for the best run. The script saves this *.csv* in the current working directory.

## Results Figures
The [resultsFigures](../scripts/resultsFigures.py) script generates three figures for the paper from the *.csv* files of results. The figures are saved in the [figures](../scripts/figures) directory.

The first figure is generated with the `success_plot()` function and is saved as [expectedTime.png](../scripts/figures/expectedTime.png). This figure shows the expected run time for the approaches with a positive success rate for each problem.

The second and third figures are generated by `time_plot()` which plots the average time taken for each model solve, separated by each approach. The second figure is saved as [costTime.png](../scripts/figures/costTime.png) and the third as [gradTime.png](../scripts/figures/gradTime.png) for solves without and with sensitivities, respectively.

## Significance
The script [significance](../scripts/significance.py) calculates the significance of the results. It uses the *.csv* files generated by [reviewOptimiserRuns](../scripts/reviewOptimiserRuns.py) and saves the significance results in new *.csv* files in the current working directory. 

The new *.csv* files contain summary information, like the expected run time, success rate, and tier, for each approach. This is followed by the bootstrapped p-value for comparing the best approach against each other approach. 

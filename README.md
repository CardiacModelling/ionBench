# ionBench
A benchmarking tool for comparing different parameter optimization algorithms for ion channel models.

## Project Structure
The tree structure of this project is outlined below. 
```
├───data
│   ├───loewe2016
│   ├───moreno2016
│   └───staircase
├───docs
├───ionBench
│   ├───benchmarker
│   ├───optimisers
│   │   ├───external_optimisers
│   │   ├───pints_optimisers
│   │   ├───scipy_optimisers
│   │   └───spsa_spsa.py
│   └───problems
└───test
```
The `data` directoy is split up into the available test problems. Each subdirectory contains the Myokit `.mmt` files, the voltage clamp protocols stored as `.csv` files, and output data to train the models, also stored as a `.csv`.

The `docs` directory is currently empty but will contain information and guides on how to use the benchmarker, the test problems and the optimisation algorithms. 

The `ionBench` directory contains the majority of the code, including the benchmarker and problems classes and the different optimisation algorithms. 

* The `benchmarker` subdirectory contains the main `Benchmarker` class that the test problems all inherit from and defines the core features of the benchmarkers. 
    
* The `optimisers` subdirectory contains all of the optimisation algorithms that are currently implemented. These are then further subdivided into three directories, containing the optimisers from pints, from scipy, and other optimisation algorithms used in fitting ion channel models that have been implemented specifically for `ionBench`. In addition to these, there is also an implementation of SPSA.
    
* Finally the `problems` directory contains the classes for the available benchmarking problems. This features the problems from Loewe et al 2016 and Moreno et al 2016 [Moreno 2016 is not yet functional]. In addition to these previously defined problems, we have introduced two further test problems, a Hodgkin-Huxley IKr model from Beattie et al 2017 and a Markov IKr model from Fink et al 2008. 
    
Finally, the `test` directory contains scripts for debugging and ensuring changes do not break previous functionality.

## Performance metrics
There are currently four optimisation algorithm performance metrics in ionBench. 

The first of these is the cost function. A good algorithm should result in a model which reproduces the data it was fitted to. This is tested by calculating the RMSE (Root Mean Squared Error) between the model predictions and the data. This is also typically the function that is minimised by these algorithms. 

The second metric is the difference between the true and estimated parameters. Reproducing the data (or producing data that is similar) is not always sufficient. Generally, we wish to correctly identify the parameters which were used to generate a synthetic data set. We quantify this by calculated the RMSE in parameter space between the estimated and true parameters. (This metric has some issues such as weighting parameters differently depending on their size. It was originally chosen when only the Staircase problems were included where all parameters are rescaled to the same size. It needs updating to a metric which is parameter scale independent). 

The third metric is the number of correctly identified parameters. It identifies how many parameters are within 5% of their true values. 

The fourth metric is the number of times the model is solved. All algorithms are able to identify the true parameters if given an infinite amount of computation time (since any algorithm can be utilised in a multistart approach). To better compare algorithms, we want to identify how fast an algorithm can identify parameters which reproduce the data or can identify the true parameters. A common metric for this is computation time however this is dependent on the machine the algorithm is run on, the precise implementation of the algorithm and other aspects that introduce noise. To resolve these issues, we track the number of times the model is solved. Solving an ODE model is by far the most computationally expensive part of ODE optimisation, where typically between 80% and 99% of the total computation time is dedicated to solving the model.

In addition to these metrics, a planned metric is first order optimality (either L2 or Linf norm of the gradient).

## The Benchmarker class
The `Benchmarker` class contains the majority of the features needed to evaluate the performance of the optimisation algorithms. Although it can be called to build a `Benchmarker` object, it is typically more useful to construct one of the test problem benchmarkers which all inherit from this class. 

The main feature of the `Benchmarker` class is it abstraction of the cost function, `benchmarker.cost()`. In addition to evaluating a RMSE cost function at the inputted parameters (`benchmarker.signedError()` and `benchmarker.squaredError()` are also available as alternative cost functions returning vectors of residuals and squared residuals, respectively), the benchmaker will also track the number of times the model has been solved, the evaluated cost at all of the inputted parameters, the RMSE in parameter space of all inputted parameters, and the number of correctly identified parameters (those within 5% of the true values) at all inputted parameters. 

Once fitting is complete, `benchmarker.evaluate()` can be called with the fitted parameters. This will report metrics like RMSE in parameter space, cost, number of identified parameters, and number of model solves. In addition, calling this function will also plot these metrics as a function of time (the order the `benchmarker.cost()` was called). Plotting can be disabled by setting `benchmarker.plotter = False`. 
Log transforms can also be specified in the benchmarker using `benchmarker.logTransform()` and inputted indexes of parameters you wish to log transform (base e), all future inputted parameters in the benchmarker can then be in log-space (or a mix if only some parameters are log transformed) while the recorded RMSE and number of identified parameters will use the original parameter space to ensure results are comparable between transformed and non-transformed optimisations.

Parameter upper and lower bounds can be included by using `benchmarker.addBounds([lb,ub])` where `lb` and `ub` are lists of lower and upper bounds. This will result in `benchmarker.cost()` returning `inf` if any of the bounds are violated. Additionally, the number of solves will not be incremented (since the time taken to compare against bounds is negligible compared with the time to solve the model) but the trackers for cost, RMSE in parameter space, and number of identified parameters will include this point. Bounds on the actual model rates are implemented in the pints optimisers but not currently in the `Benchmarker` class. Interior point style bounds (using a barrier function to have continuous and differentiable bounds) will also be impletented in future so bounds can be used with gradient-based methods.
Other useful function include:

* `benchmarker.reset()` which resets the simulation object and the trackers (solve count, costs, parameter RMSEs, number of identified parameters) without the need to recompile the Myokit model. 
* `benchmarker.n_parameters()` which returns the number of parameters in the model.

## Problems
### Staircase
There are two Staircase test problems. The first is a Hodgkin-Huxley IKr model from Beattie et al and the second is a Markov IKr model from Fink et al 2008. Both models use the staircase protocol and fit the resulting current trace. These models can be instantiated as given below:
```
import ionBench
bm1 = ionBench.problems.staircase.HH_Benchmarker()
bm2 = ionBench.problems.staircase.MM_Benchmarker()
```
The parameters used in simulating these models are scaling factors, applied to the default rates (`benchmarker.defaultParams`), as such the default values or initial guesses should be a vector of all ones, or sampled by randomly perturbing a vector of all ones.

Additionally, there is a `generateData()` function, which takes inputs of `HH` for Hodgkin-Huxley or `MM` for Markov Model, and randomly samples parameters (perturbing the default rates to random values between 50% and 150%). This function then simulates the model with these rates, storing the current trace in either `dataHH.csv` or `dataMM.csv` and the scaling factors on the parameters in either `trueParamsHH.csv` or `trueParamsMM.csv`, both kept in the `data/staircase` directory.

The Staircase problems are also the only problems in the benchmarker to use noisy (synthetic) data.

### Loewe 2016
Four of the problems from Loewe et al 2016 are implemented in `ionBench`. These are IKr and IKur (defined in the paper as "easy" and "hard", respectively), both with the narrow and wide parameter spaces. For these test problems, the models are run using a step protocol (-80mV followed by a varying step height between +50mV and -70mV for 400ms and a step down to -110mV) and the models are fitted to the resulting current trace. These benchmarkers can be instantiated as given below:
```
import ionBench
bm1 = ionBench.problems.loewe2016.ikr()
bm2 = ionBench.problems.loewe2016.ikur()
```
The parameters used in the model are as originally described in Loewe et al 2016 (not scaling factors). These benchmarkers feature a sampling function, `benchmarker.sample()`, which samples from the parameter search space defined in Loewe et al 2016. By default it samples from the narrow space but this can be changed by setting `benchmarker.paramSpaceWidth = 2` to sample from the wide parameter space. 

There is also a `loewe2016.generateData()` function which takes inputs of `ikr` or `ikur` and generates the data files `data/loewe2016/ikr.csv` or `data/loewe2016/ikur.csv`, respectively. These data files are generated using the default parameters, as described in Loewe et al 2016. These parameters are also the center of the parameter search space, this means starting points for optimisation should always be from sampled rates and optimisation algorithms that average over multiple sampled parameters should not be tested using this problem. 

### Moreno 2016
Note: This test problem does not work yet.

This problem uses the INa model described in Moreno et al 2016 and fits to a variety of summary curves. The benchmarker can be instantiated as given below:
```
import ionBench
bm = ionBench.problems.moreno2016.ina()
```
The parameters used in the model are as originally described in Moreno et al 2016 (not scaling factors). This benchmarkers feature a sampling function, `benchmarker.sample()`, which perturbs the default parameters between 95% and 105% (an optional parameter width can be used to vary the range of the perturbation [default=5] with `width=5`, `width=10`, and `width=25` being used in Moreno et al 2016). The data file `data/moreno2016/ina.csv` contains the summary curve values to which the model is fitted. This file is generated using the `benchmarker.generateData()` function. As was the case with the loewe2016 benchmarkers, the data is generated using the default parameters so algorithms should be started at sampled parameters (using `benchmarker.sample()`) and algorithms which average over sampled parameters should not be used with this problem.

## Optimisation Algorithms
There are a wide variety of optimisation algorithms used for ion channel models. We have implementations of optimisation algorithms from Pints, Scipy, and paper specific algorithms. Each algorithm can be run either as a script, in which case it is used to fit the Hodgkin-Huxley Staircase problem, or as a function taking a benchmarker object as an input.

### Pints
The Pints optimisers currently available are CMA-ES, Nelder-Mead, PSO, SNES, and XNES. 

CMA-ES features bounds on both the rates and the parameters, defined in the pints_optimisers.classes_pints.AdvancedBoundaries, and log transforms on the rates (defined through Pints rather than ionBench). The upper and lower bounds on the parameters and which parameters should be log-transformed are both customisable but the bounds on the transition rates are currently hard-coded. The maximum number of iterations in CMA-ES can be defined by setting the variable `maxIter`, and CMA-ES will be restarted at random starting parameters `iterCount` times. Example code to run CMA-ES is given below:
```
import ionBench
bm = ionBench.problems.staircase.HH_Benchmarker()
logTransforms = [0, 2, 4, 6]
localBounds = [[0,1e-7,1e3], [1,1e-7,0.4], [2,1e-7,1e3], [3,1e-7,0.4], [4,1e-7,1e3], [5,1e-7,0.4], [6,1e-7,1e3], [7,1e-7,0.4]]
kCombinations = [[0,1], [2,3], [4,5], [6,7]]
optimisedParameters = ionBench.optimisers.pints_optimisers.cmaes_pints.run(bm = bm, kCombinations = kCombinations, localBounds = localBounds, logTransforms = logTransforms)
```

The other Pints optimisers all work similarly, but do not include bounds.

### Scipy
The Scipy optimisers currently available are lm (Levenberg-Marquardt), Nelder-Mead, Powell's simplex method, and Trust Region Reflective. 

Nelder-Mead can be run using the following code:
```
import ionBench
bm = ionBench.problems.staircase.HH_Benchmarker()
bounds = [(0,None)]*bm.n_parameters()
optimisedParameters = ionBench.optimisers.scipy_optimisers.nelderMead_scipy.run(bm = bm, x0 = bm.sample(), bounds = bounds)
```

The other Scipy optimisers all work similiarly, however lm does not include bounds as an option and the form of the bounds varies between the algorithms (matching the form passed into scipy).

### External Optimisers
We currently have four optimisers defined from other papers for ion channel fitting. Two of these are the genetic algorithms from Bot et al 2012 (GA_Bot2012) and Smirnov et al 2020 (GA_Smirnov2020), one is the pattern search algorithm defined in Kohjitani et al 2022 (patternSearch_Kohjitani2022), and the final algorithm is perturbed particle swarm optimisation defined in Chen et al 2012 (ppso_Chen2012).

Because all these algorithms come with defined default values for hyper-parameters, they can all be run using only a benchmarker:
```
import ionBench
bm = ionBench.problems.staircase.HH_Benchmarker()
optimisedParameters_bot = ionBench.optimisers.external_optimisers.GA_Bot2012.run(bm)
optimisedParameters_smirnov = ionBench.optimisers.external_optimisers.GA_Smirnov2020.run(bm)
optimisedParameters_smirnov = ionBench.optimisers.external_optimisers.patternSearch_Kohjitani2022.run(bm)
optimisedParameters_smirnov = ionBench.optimisers.external_optimisers.ppso_Chen2012.run(bm)
```

### SPSA
Finally, we have an implementation of SPSA (Simultaneous Perturbation Stochastic Approximation) using the `spsa` python package. This currently converges very slowly so either needs tuning in the hyper-parameters or a new version needs to be written. It can be run with the following code:
```
import ionBench
bm = ionBench.problems.staircase.HH_Benchmarker()
x0 = bm.defaultParams
ionBench.optimisers.spsa_spsa.run(bm, x0)
```

## Future Features
* Bounds - Current only bounds on the parameters can be included in the benchmarker but it would be nice to have bounds on the rates. Additionally, it would be good to include barrier function style bounds to allow them to work nicely with gradient based methods.

* Additional optimisation algorithms - There are still some scipy optimisers to add, some of the pints optimisers require derivatives and it would be nice to add a function for this into the benchmarker so it is simpler for the user to include similar algorithms. Additionally, there are still lots of different algorithms from various papers for fitting ion channel models to include (Current plans to include a further 19 external optimisers). 

* Analytical solvers - Both the Moreno 2016 and Loewe 2016 problems could use analytical solvers. This would fix some of the issues in choosing solver tolerances for these problems and significantly speed them up.

* Parallelisation - Its not clear yet how well the benchmarker would handle being run in parallel (specifically for the tracker) but it is something that would be worth looking into.

* Real data - Both Moreno et al 2016 and Loewe et al 2016 include real data in the papers. It would be nice to see how the algorithms handle fitting to real data but its not clear how to best implement the performance metrics, two of which rely on knowing the true parameters.

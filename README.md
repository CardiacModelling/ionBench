# ionBench
A benchmarking tool for comparing different parameter optimization algorithms for ion channel models.

## Project Structure
The tree structure of this project is outlined below. 
```

├───docs
├───ionbench
│   ├───benchmarker
│   ├───data
│   │   ├───loewe2016
│   │   ├───moreno2016
│   │   └───staircase
│   ├───optimisers
│   │   ├───external_optimisers
│   │   ├───pints_optimisers
│   │   ├───scipy_optimisers
│   │   └───spsa_spsa.py
│   └───problems
└───test
```
The __data__ directoy is split up into the available test problems. Each subdirectory contains the Myokit *.mmt* files, the voltage clamp protocols stored as *.csv* files where relevant, and output data to train the models, also stored as a *.csv*.

The __docs__ directory is currently empty but will contain information and guides on how to use the benchmarker, the test problems and the optimisation algorithms. 

The __ionbench__ directory contains the majority of the code, including the benchmarker and problems classes and the different optimisation algorithms. 

* The __benchmarker__ subdirectory contains the main __Benchmarker__ class that the test problems all inherit from and defines the core features of the benchmarkers. It also contains the __Tracker__ class which contains the functions to take performance metrics over time.
    
* The __optimisers__ subdirectory contains all of the optimisation algorithms that are currently implemented. These are then further subdivided into three directories, containing the optimisers from ***pints***, from ***scipy***, and other optimisation algorithms used in fitting ion channel models that have been implemented specifically for ***ionBench***. In addition to these, there is also an implementation of SPSA.
    
* Finally the __problems__ directory contains the classes for the available benchmarking problems. This features the problems from Loewe et al 2016 and Moreno et al 2016. In addition to these previously defined problems, we have introduced two further test problems, a Hodgkin-Huxley IKr model from Beattie et al 2017 and a Markov IKr model from Fink et al 2008. 
    
Finally, the __test__ directory contains scripts for debugging and ensuring changes do not break previous functionality.

## Performance metrics
There are currently four optimisation algorithm performance metrics in ***ionBench***. 

The first of these is the cost function. A good algorithm should result in a model which reproduces the data it was fitted to. This is tested by calculating the RMSE (Root Mean Squared Error) between the model predictions and the data. This is also typically the function that is minimised by these algorithms. 

The second metric is the difference between the true and estimated parameters. Reproducing the data (or producing data that is similar) is not always sufficient. Generally, we wish to correctly identify the parameters which were used to generate a synthetic data set. We quantify this by calculated the RMSRE (Root Mean Squared Relative Error) in parameter space between the estimated and true parameters. 

The third metric is the number of correctly identified parameters. It identifies how many parameters are within 5% of their true values. 

The fourth metric is the number of times the model is solved. All algorithms are able to identify the true parameters if given an infinite amount of computation time (since any algorithm can be utilised in a multistart approach). To better compare algorithms, we want to identify how fast an algorithm can identify parameters which reproduce the data or can identify the true parameters. A common metric for this is computation time. However, this is dependent on the machine the algorithm is run on, the precise implementation of the algorithm and other aspects that introduce noise. To resolve these issues, we track the number of times the model is solved. Solving an ODE model is by far the most computationally expensive part of ODE optimisation, where typically between 80% and 99% of the total computation time is dedicated to solving the model.

In addition to these metrics, a planned metric is first order optimality (either L2 or Linf norm of the gradient).

## The Benchmarker class
The __Benchmarker__ class contains the majority of the features needed to evaluate the performance of the optimisation algorithms. Although it can be called to build a __Benchmarker__ object, it is typically more useful to construct one of the test problem benchmarkers which all inherit from this class. 

The main feature of the __Benchmarker__ class is it abstraction of the cost function, `benchmarker.cost()`. In addition to evaluating a RMSE cost function at the inputted parameters (`benchmarker.signed_error()` and `benchmarker.squared_error()` are also available as alternative cost functions, returning vectors of residuals and squared residuals, respectively). Through the use of the __Tracker__ class, the benchmaker will also record the number of times the model has been solved, the evaluated cost at all of the inputted parameters, the RMSRE in parameter space of all inputted parameters, and the number of correctly identified parameters (those within 5% of the true values) at all evaluated parameters. 

Once fitting is complete, `benchmarker.evaluate()` can be called with the fitted parameters. This will report the performance metrics like RMSRE in parameter space, cost, number of identified parameters, and number of model solves. In addition, calling this function will also plot these metrics as a function of time (the order the `benchmarker.cost()` was called). Plotting can be disabled by setting `benchmarker.plotter = False`. 

Log transforms can also be specified in the benchmarker using `benchmarker.log_transform()` and inputting indexes of parameters you wish to log transform (base e), all future inputted parameters in the benchmarker can then be in log-space (or a mix if only some parameters are log transformed), while the recorded RMSRE and number of identified parameters will use the original parameter space to ensure results are comparable between transformed and non-transformed optimisations.

When working with log transforms (or equivalenty scale factor transforms using `benchmarker._useScaleFactors=True`), it can be useful to use the functions `benchmarker.input_parameter_space()` and `benchmarker.original_parameter_space()` for transforming parameters. 

Parameter upper and lower bounds can be included by using `benchmarker.add_bounds([lb,ub])` where `lb` and `ub` are lists of lower and upper bounds. This will result in `benchmarker.cost()` returning `inf` if any of the bounds are violated. Additionally, the number of solves will not be incremented (since the time taken to compare against bounds is negligible compared with the time to solve the model) but the trackers for cost, RMSE in parameter space, and number of identified parameters will include this point. Bounds on the actual model rates are implemented in the ***pints*** optimisers but not currently in the __Benchmarker__ class. Interior point style bounds (using a barrier function to have continuous and differentiable bounds) will also be impletented in future so bounds can be used with gradient-based methods.

Other useful functions include:

* `benchmarker.reset()` which resets the simulation object and the trackers (solve count, costs, parameter RMSEs, number of identified parameters) without the need to recompile the Myokit model. 
* `benchmarker.n_parameters()` which returns the number of parameters in the model.

## Problems
All problems feature a `generate_data()` method to construct the synthetic datasets, and a `benchmarker.sample()` function for randomly sampling parameters. In all problems, the `benchmarker.defaultParams` are the same as the `benchmarker._trueParams`, so averaging in parameter space should be avoided, and initial points should be random and defined by `benchmarker.sample()`

### Staircase
There are two Staircase test problems. The first is a Hodgkin-Huxley IKr model from Beattie et al and the second is a Markov IKr model from Fink et al 2008. Both models use the staircase protocol and fit the resulting current trace. These models can be instantiated as given below:
```
import ionbench
bm1 = ionbench.problems.staircase.HH_Benchmarker()
bm2 = ionbench.problems.staircase.MM_Benchmarker()
```
The parameters used in simulating these models are scaling factors, applied to the default rates (`benchmarker.defaultParams`), as such the default values should be a vector of all ones, or sampled by randomly perturbing a vector of all ones.

The `benchmarker.sample()` method randomly perturbs the true parameters by +-50%.

The Staircase problems are also the only problems in the benchmarker to use noisy data for fitting.

### Loewe 2016
Four of the problems from Loewe et al 2016 are implemented in ***ionBench***. These are IKr and IKur (defined in the paper as "easy" and "hard", respectively), both with the narrow and wide parameter spaces. For these test problems, the models are run using a step protocol (-80mV followed by a varying step height between +50mV and -70mV for 400ms and a step down to -110mV) and the models are fitted to the resulting current trace. These benchmarkers can be instantiated as given below:
```
import ionbench
bm1 = ionbench.problems.loewe2016.ikr()
bm2 = ionbench.problems.loewe2016.ikur()
```
By default `benchmarker.sample()` samples from the narrow space but this can be changed by setting `benchmarker.paramSpaceWidth = 2` to sample from the wide parameter space. Each parameter is either defined as multiplicative, in which case it is sampled between x0.1 and x10 in a log-uniform distribution, or as additive, in which case it is sampled between +-60, with these ranges doubling for the wide parameter space. 

One of the parameters in IKur has a true value of 0.005 and is additive, so is sampled between -59.995 and 60.005, which results in very large values of RMSRE in parameter space compared with the other models.

### Moreno 2016
This problem uses the INa model described in Moreno et al 2016 and fits to a variety of summary curves. The benchmarker can be instantiated as given below:
```
import ionbench
bm = ionbench.problems.moreno2016.ina()
```
The parameters used in the model are as originally described in Moreno et al 2016. This benchmarkers feature a sampling function, `benchmarker.sample()`, which perturbs the default parameters between 95% and 105% (an optional parameter width can be used to vary the range of the perturbation (default=5) with `width=5`, `width=10`, and `width=25` being used in Moreno et al 2016). The data file *data/moreno2016/ina.csv* contains the summary curve values to which the model is fitted. This file is generated using the `benchmarker.generate_data()` function. 

## Optimisation Algorithms
There are a wide variety of optimisation algorithms used for ion channel models. We have implementations of optimisation algorithms from Pints, Scipy, and paper specific algorithms. Each algorithm can be run either as a script, in which case it is used to fit the Hodgkin-Huxley Staircase problem, or as a function taking a benchmarker object as an input.

All optimisers will make use of bounds defined in the benchmarker where applicable.

Initial points can be specified in the optimisation algorithms, but if none are given, the optimisers will use the benchmarkers `.sample()` method.

### Pints
The Pints optimisers currently available are CMA-ES, Nelder-Mead, PSO, SNES, and XNES. 

CMA-ES features bounds on both the rates and the parameters, defined in the __ionbench.optimisers.pints_optimisers.classes_pints.AdvancedBoundaries__, and log transforms on the rates (defined through ***pints*** rather than ***ionBench***). The upper and lower bounds on the parameters and which parameters should be log-transformed are both customisable but the bounds on the transition rates are currently hard-coded. The maximum number of iterations in CMA-ES can be defined by setting the variable `maxIter`. Example code to run CMA-ES is given below:
```
import ionbench
bm = ionbench.problems.staircase.HH_Benchmarker()
bm.log_transform([True, False]*4+[False])
bm.add_bounds([[1e-7]*8+[-np.inf], [1e3,0.4]*4+[np.inf]])
ionbench.optimisers.pints_optimisers.cmaes_pints.run(bm)
```

The other Pints optimisers all work similarly.

### Scipy
The Scipy optimisers currently available are lm (Levenberg-Marquardt), Nelder-Mead, Powell's simplex method, and Trust Region Reflective. 

Nelder-Mead can be run using the following code:
```
import ionbench
bm = ionbench.problems.staircase.HH_Benchmarker()
bm.add_bounds([[0]*9, [np.inf]*9])
optimisedParameters = ionbench.optimisers.scipy_optimisers.nelderMead_scipy.run(bm)
```

The other Scipy optimisers all work similiarly, however lm does not include bounds as an option and the form of the bounds varies between the algorithms (matching the form passed into ***scipy***).

### External Optimisers
We currently have five optimisers defined from other papers for ion channel fitting. Two of these are the genetic algorithms from Bot et al 2012 (GA_Bot2012) and Smirnov et al 2020 (GA_Smirnov2020), one is the pattern search algorithm defined in Kohjitani et al 2022 (patternSearch_Kohjitani2022), the perturbed particle swarm optimisation defined in Chen et al 2012 (ppso_Chen2012), and the hybrid PSO+TRR algorithm from Loewe et al 2016.

```
import ionbench
bm = ionbench.problems.staircase.HH_Benchmarker()
optimisedParameters_bot = ionbench.optimisers.external_optimisers.GA_Bot2012.run(bm)
optimisedParameters_smirnov = ionbench.optimisers.external_optimisers.GA_Smirnov2020.run(bm)
optimisedParameters_kohjitani = ionbench.optimisers.external_optimisers.patternSearch_Kohjitani2022.run(bm)
optimisedParameters_chen = ionbench.optimisers.external_optimisers.ppso_Chen2012.run(bm)
optimisedParameters_loewe = ionbench.optimisers.external_optimisers.hybridPSOTRR_Loewe2016.run(bm)
```

### SPSA
Finally, we have an implementation of SPSA (Simultaneous Perturbation Stochastic Approximation) using the ***spsa*** python package. This currently converges very slowly so either needs tuning in the hyper-parameters or a new version needs to be written. It can be run with the following code:
```
import ionbench
bm = ionbench.problems.staircase.HH_Benchmarker()
ionbench.optimisers.spsa_spsa.run(bm)
```

## Workflow
The intended workflow for using the benchmarker is to generate a benchmarker object, make any changes such as log transforms or bounds (either on parameters or on rates), and pass it into an optimisation algorithm to evaluate. There should be minimal differences between the inputs for the optimisation algortithms, particularly in required inputs.
```
import numpy as np
import ionbench
bm = ionbench.problems.staircase.MM_Benchmarker()
bm.add_bounds([[0]*bm.n_parameters(),[np.inf]*bm.n_parameters()])
bm.log_transform([True, False, True]*2+[False]+[True, False, True]*2+[False]*2)
optimisedParameters = ionbench.optimisers.scipy_optimisers.nelderMead_scipy.run(bm)
```

## Future Features
* Bounds - Current only bounds on the parameters can be included in the benchmarker but it would be nice to have bounds on the rates. Additionally, it would be good to include barrier function style bounds to allow them to work nicely with gradient based methods.

* Additional optimisation algorithms - There are still some ***scipy*** optimisers to add, some of the ***pints*** optimisers require derivatives and it would be nice to add a function for this into the benchmarker so it is simpler for the user to include similar algorithms. Additionally, there are still lots of different algorithms from various papers for fitting ion channel models to include (Current plans to include a further 19 external optimisers). 

* Parallelisation - Its not clear yet how well the benchmarker would handle being run in parallel (specifically for the tracker) but it is something that would be worth looking into.

* Real data - Both Moreno et al 2016 and Loewe et al 2016 include real data in the papers. It would be nice to see how the algorithms handle fitting to real data but its not clear how to best implement the performance metrics, two of which rely on knowing the true parameters.
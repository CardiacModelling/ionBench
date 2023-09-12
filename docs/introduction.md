# Introduction
This guide walks what ***ionBench*** is and does. It gives an overview of the theory side of the module. If you want more pratical information such as clear steps on how to use the module, check out the __tutorial__.

## Performance metrics
There are currently four performance metrics in ***ionBench***. Each metric measures a particular aspect of algorithm performance.

The first of these is the cost function. A good algorithm should result in a model which reproduces the data it was fitted to. This is tested by calculating the RMSE (Root Mean Squared Error) between the model predictions and the data. This is also typically the function that is minimised by these optimisers. 

The second metric is the difference between the true and estimated parameters. Reproducing the data (or producing data that is similar) is not always sufficient. Generally, we wish to correctly identify the parameters which were used to generate a synthetic data set. We quantify this by calculating the RMSRE (Root Mean Squared Relative Error) in parameter space between the estimated and true parameters. 

The third metric is the number of correctly identified parameters. It identifies how many parameters are within 5% of their true values. 

The fourth metric is the number of times the model is solved. All algorithms are able to identify the true parameters if given an infinite amount of computation time (since any algorithm can be utilised in a multistart approach). To better compare algorithms, we want to identify how fast an algorithm can identify parameters which reproduce the data or can identify the true parameters. A common metric for this is computation time. However, this is dependent on the machine the algorithm is run on, the precise implementation of the algorithm and other aspects that introduce noise. To resolve these issues, we track the number of times the model is solved. Solving an ODE model is by far the most computationally expensive part of ODE optimisation, where typically between 80% and 99% of the total computation time is dedicated to solving the model.

In addition to these metrics, a planned metric is first order optimality (either L2 or Linf norm of the gradient).

These performance metrics are tracked automatically when using ***ionBench***. They are stored in a __Tracker__ object associated with each benchmarker problem class. 

## The Benchmarker class
The __Benchmarker__ class contains the majority of the features needed to evaluate the performance of the optimisation algorithms. A __Benchmarker__ object should only be used as part of the construction of a specific problem. Each problem has its own __Benchmarker__ class that inherits all of the methods and data from the main __Benchmarker__ class but also contians lots of problem-specific information. 

The main feature of the __Benchmarker__ classes are there abstraction of the cost function, `benchmarker.cost(parameters)`. This evalutes a RMSE cost function at the inputted parameters compared with the __Benchmarkers__ pregenerated synthetic data (`benchmarker.signed_error(parameters)` and `benchmarker.squared_error(parameters)` are also available as alternative cost functions, returning vectors of residuals and squared residuals, respectively). Through the use of the __Tracker__ class, the benchmaker will also record the number of times the model has been solved, the evaluated cost at all of the inputted parameters, the RMSRE in parameter space of all inputted parameters, and the number of correctly identified parameters (those within 5% of the true values) at all evaluated parameters. 

Once fitting is complete, `benchmarker.evaluate(parameters)` can be called with the fitted parameters. This will report the performance metrics like RMSRE in parameter space, cost, number of identified parameters, and number of model solves. In addition, calling this function will also plot these metrics as a function of time (the order the `benchmarker.cost()` was called). Plotting can be disabled by setting `benchmarker.plotter = False`. 

Log transforms can also be specified in the benchmarker using `benchmarker.log_transform()` and inputting indexes of parameters you wish to log transform (base e), all future inputted parameters in the benchmarker can then be in log-space (or a mix if only some parameters are log transformed), while the recorded RMSRE and number of identified parameters will use the original parameter space to ensure results are comparable between transformed and non-transformed optimisations.

When working with log transforms (or equivalenty scale factor transforms using `benchmarker._useScaleFactors=True`), it can be useful to use the functions `benchmarker.input_parameter_space()` and `benchmarker.original_parameter_space()` for transforming parameters. 

Parameter upper and lower bounds can be included by using `benchmarker.add_bounds([lb,ub])` where `lb` and `ub` are lists of lower and upper bounds. This will result in `benchmarker.cost()` returning `inf` if any of the bounds are violated. Additionally, the number of solves will not be incremented (since the time taken to compare against bounds is negligible compared with the time to solve the model) but the trackers for cost, RMSE in parameter space, and number of identified parameters will include this point. More advanced options for bounds are still to come.

Other useful functions include:

* `benchmarker.reset()` which resets the simulation object and the __Tracker__ (solve count, costs, parameter RMSEs, number of identified parameters) without the need to recompile the Myokit model. 
* `benchmarker.n_parameters()` which returns the number of parameters in the model.

## Problems
There are seven benchmarking problems currently in ***ionBench***. These are four problems from Loewe et al 2016, one from Moreno et al 2016 and two developed specifically for ***ionBench***. Each problem has an associated __Benchmarker__ class that inherits from the main __Benchmarker__ class described above. The problems store general information like log-transform status or parameter bounds, problem-specific information such as a problems Myokit model, the synthetic data and a parameter sampling method `benchmarker.sample()`, and run-specific information such as the __Tracker__ object which stores the evaluated performance metrics over time. 

Each problem also has an associated `generate_data()` method which will simulate the model and store the model output data in the __data__ directory. 

### Loewe 2016
Four of the problems from Loewe et al 2016 are implemented in ***ionBench***. These are IKr and IKur (defined in the paper as "easy" and "hard", respectively), both with the narrow and wide parameter spaces. For these test problems, the models are run using a step protocol (-80mV followed by a varying step height between +50mV and -70mV for 400ms and a step down to -110mV) and the models are fitted to the resulting current trace.

By default `benchmarker.sample()` samples from the narrow space but this can be changed by setting `benchmarker.paramSpaceWidth = 2` to sample from the wide parameter space. Each parameter is either defined as multiplicative, in which case it is sampled between x0.1 and x10 in a log-uniform distribution, or as additive, in which case it is sampled between +-60, with these ranges doubling (x0.01 and x100 in the case of multiplicative parameters) for the wide parameter space. 

One of the parameters in IKur has a true value of 0.005 and is additive, so is sampled between -59.995 and 60.005, which results in very large values of RMSRE in parameter space compared with the other models.

### Moreno 2016
This problem uses the INa model described in Moreno et al 2016 and fits to a variety of summary curves measuring steady state activation and inactivation, recovery from inactivation, recovery from use-dependent block and time to decay 50%. 

The parameters used in the model are as originally described in Moreno et al 2016. This benchmarker's sampling function perturbs the true parameters between 95% and 105% (as with Loewe 2016, the width of this interval can be changed by setting `benchmarker.paramSpaceWidth`, with 5 (default), 10, and 25 being used in Moreno et al 2016). 

### Staircase
There are two Staircase test problems developed for ***ionBench***. The first is a Hodgkin-Huxley IKr model from Beattie et al 2017 and the second is a Markov IKr model from Fink et al 2008. Both models use the staircase protocol and fit the resulting current trace. 

The `benchmarker.sample()` method randomly perturbs the true parameters by +-50%.

The Staircase problems are also the only problems in the benchmarker to use noisy data for fitting.



## Optimisation Algorithms
There are a wide variety of optimisation algorithms used for ion channel models. We have implementations of optimisation algorithms from Pints, Scipy, and paper specific algorithms impleted for ***ionBench***. All optimisers only require a benchmarker as an input, with hyperparameters and initial guesses being optional inputs (`benchmarker.sample()` is used if no initial guess is specified). 

Optimisers will automatically load parameter bounds from the benchmarker if they can be used in the algorithm. Each optimiser module has a `.run(bm=benchmarker)` method that runs the optimisation on a benchmarker.


### Pints
The Pints optimisers currently available are CMA-ES, Nelder-Mead, PSO, SNES, and XNES. 

For CMA-ES, if the benchmarker has active bounds, then CMA-ES will automatically use bound on the transition rates in addition to parameter bounds, defined in the __ionbench.optimisers.pints_optimisers.classes_pints.AdvancedBoundaries__ module. 

### Scipy
The Scipy optimisers currently available are LM (Levenberg-Marquardt), Nelder-Mead, Powell's simplex method, and Trust Region Reflective. 

Both Trust Region Reflective and LM make use of the alternative cost functions, in this case `benchmarker.signed_error()` returning a vector of residuals rather than the RMSE cost. 

### External Optimisers
We currently have five optimisers defined from other papers for ion channel fitting. Two of these are the genetic algorithms from Bot et al 2012 (GA_Bot2012) and Smirnov et al 2020 (GA_Smirnov2020), one is the pattern search algorithm defined in Kohjitani et al 2022 (patternSearch_Kohjitani2022), the perturbed particle swarm optimisation defined in Chen et al 2012 (ppso_Chen2012), and the hybrid PSO+TRR algorithm from Loewe et al 2016 (hybridPSOTRR_Loewe2016).

### SPSA
Finally, we have an implementation of SPSA (Simultaneous Perturbation Stochastic Approximation) using the ***spsa*** python package. This currently converges very slowly so either needs tuning in the hyper-parameters or a new version needs to be written. 

## Approaches
Some optimisation algorithms come with recommendations for other characteristics such as log-transforms or bounds to be applied to the parameters. These are implemented in ***ionBench*** through the use of approaches. Approaches store benchmarker settings, such as to log-transform parameters, add bounds, or use scale factors, that can then be applied to any benchmarker problem object. 

They can be particularly useful if you want to apply problem-specific settings, such as bounds determined by the problem-specific sampling function. In addition to constructing a new approach, standard approach defined by different papers can also be loaded.

Each optimiser has a `.get_approach()` function which will get the recommended approach for that particular optimiser.

## Uncertainty
There are two uncertainty and unindentifiability tools built into ***ionBench***. The first is a profile likelihood calculator, which will generate, plot and save profile likelihood plots for the inputted benchmarker problem. The second is a Fisher's Information Matrix calculator. This uses the curvature of the likelihood to find the FIM but it will likely be switched over to a MCMC approach at some point so its a bit more reliable. 
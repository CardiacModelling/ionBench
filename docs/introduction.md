# Introduction
This guide walks what ***ionBench*** is and does. It gives an overview of the theory side of the module. If you want more practical information such as clear steps on how to use the module, check out the __tutorial__.

## Performance metrics
There are currently four performance metrics in ***ionBench***. Each metric measures a particular aspect of algorithm performance.

The first of these is the cost function. A good algorithm should result in a model which reproduces the data it was fitted to. This is tested by calculating the RMSE (Root Mean Squared Error) between the model predictions and the data. This is also typically the function that is minimised by these optimisers. 

The second metric is the time spent solving the model. All algorithms are able to identify the true parameters if given an infinite amount of computation time (since any algorithm can be utilised in a multistart approach). To better compare algorithms, we want to identify how fast an algorithm can identify parameters which reproduce the data. We track the number of model solves, whether these solves were the more expensive 'with sensitivities' versions for calculating gradients, and the time for each solve.

These performance metrics are tracked automatically when using ***ionBench***. They are stored in a __Tracker__ object associated with each benchmarker problem. 

## The Benchmarker class
The __Benchmarker__ class contains the majority of the features needed to evaluate the performance of the optimisation algorithms. A __Benchmarker__ object should only be used as part of the construction of a specific problem. Each problem has its own __Benchmarker__ subclass that inherits all the methods and data from the __Benchmarker__ class but also contains lots of problem-specific information. 

The main feature of the __Benchmarker__ classes are their abstraction of the cost function, `benchmarker.cost(parameters)`. This evaluates a RMSE cost function at the inputted parameters compared with the __Benchmarkers__ pregenerated synthetic data (`benchmarker.signed_error(parameters)` and `benchmarker.squared_error(parameters)` are also available as alternative cost functions, returning vectors of residuals and squared residuals, respectively). Through the use of the __Tracker__ class, the benchmarker will also record the number of times the model has been solved, the time spent on each solve, the evaluated cost at all the inputted parameters, all evaluated parameters and whether they were solved with or without sensitivities.

In addition to evaluating the cost of a set of parameters, the benchmarkers can also evaluate the gradient of the cost function at a set of parameters. This can be done by calling `benchmarker.grad(parameters)` and can also be used to find the jacobian of the signed error.

Once fitting is complete, `benchmarker.evaluate()` will be called. This will report the performance metrics like best cost and number of model solves and time spent solving the model. In addition, calling this function will also plot some of these metrics over the course of the optimisation, a plot of the model output compared with the data, and histograms of the time to solve the model. Plotting can be disabled by setting `benchmarker.plotter = False`. 

Log transforms can also be specified in the benchmarker using `benchmarker.log_transform()` and inputting a list of booleans indicating which parameters you wish to log transform (base e), all future inputted parameters in the benchmarker will then be interpreted in log-space (or a mixed space if only some parameters are log transformed). The information stored in the __Tracker__ does not use any transforms.

When working with log transforms (or equivalently scale factor transforms using `benchmarker.useScaleFactors=True`), it can be useful to use the functions `benchmarker.input_parameter_space()` and `benchmarker.original_parameter_space()` for transforming parameters. 

Parameter upper and lower bounds can be included by using `benchmarker.add_parameter_bounds()`. This will set `bm.lb=bm._LOWER_BOUND`, `bm.ub=bm._UPPER_BOUND` and `benchmarker.parametersBounded=True`. If an optimisers tries to find the cost of a parameter outside of these bounds, the model will not be solved and the cost will be given by the penalty function. The __Tracker__ will record these parameters but remember that the model was not solved.

Similarly, rate bounds can be added with `benchmarker.add_rate_bounds()`.

Other useful functions include:

* `benchmarker.reset()` which resets the simulation object and the __Tracker__ (solve count, costs, parameter RMSEs, number of identified parameters) without the need to recompile the Myokit model. It also resets any bounds and transforms, although this can be changed by passing `full=False`.
* `benchmarker.n_parameters()` which returns the number of parameters in the model.

## Problems
There are five benchmarking problems currently in ***ionBench***. These are two problems from Loewe et al. 2016, one from Moreno et al. 2016 and two developed specifically for ***ionBench***, one of which is based on Clerx et al. 2019. Each problem has an associated __Benchmarker__ class that inherits from the main __Benchmarker__ class described above. The problems store general information like log-transform status or parameter bounds, problem-specific information such as a problems Myokit model, the synthetic data and a parameter sampling method `benchmarker.sample()`, and run-specific information such as the __Tracker__ object which stores the evaluated performance metrics over time. 

Each problem also has an associated `generate_data()` method which will simulate the model and store the model output data in the __data__ directory. 

### Loewe 2016
Two of the problems from Loewe et al. 2016 are implemented in ***ionBench***. These are IKr and IKur (defined in the paper as "easy" and "hard", respectively). For these problems, the models are run using a step protocol (-80mV followed by a varying step height between +50mV and -70mV for 400ms and a step-down to -110mV) and the models are fitted to the resulting current trace.

Each parameter is either defined as multiplicative, in which case it is sampled between x0.1 and x10 in a log-uniform distribution, or as additive, in which case it is sampled between +-60.

### Moreno 2016
This problem uses the INa model described in Moreno et al. 2016 and fits to a variety of summary curves measuring steady state activation and inactivation, recovery from inactivation, and time to decay 50%. While the problem implemented in ***ionBench*** is based off the problem described in Moreno et al. 2016, our implementation measures fewer summary statistics in order to shorten the protocol and therefore the time spent solving the model.

The parameters and model used in the model are as originally described in Moreno et al. 2016. This benchmarker's sampling function perturbs the true parameters between 25% and 125%.

### Staircase
There are two Staircase problems developed for ***ionBench***. The first is a Hodgkin-Huxley IKr model from Beattie et al. 2017, and the second is a Markov IKr model from Fink et al. 2008. Both models use the staircase protocol and fit the resulting current trace. 

The `benchmarker.sample()` method samples parameters from within the parameter and rate bounds, sampling in a log transformed parameter space for some parameters. If parameters are evaluated outside the parameter or rate bounds, the penalty function is applied, regardless of if that approach used parameter or rate bounds. 

The Staircase problems are also the only problems in the benchmarker to use noisy data for fitting.

## Optimisation Algorithms
There are a wide variety of optimisation algorithms used for ion channel models. We have implementations of optimisation algorithms from Pints, Scipy, and paper specific algorithms implemented for ***ionBench***. All optimisers only require a benchmarker as an input, with hyperparameters and an initial parameter guess being optional inputs (`benchmarker.sample()` is used if no initial guess is specified). 

Optimisers will automatically load parameter bounds from the benchmarker if they can be used in the algorithm. Each optimiser module has a `.run(bm=benchmarker)` method that runs the optimisation on a benchmarker.


### Pints
The Pints optimisers currently available are CMA-ES, Nelder-Mead, PSO, rProp, SNES, and XNES. 

The Pints optimisers are the only ones which can incorporate rate bounds into the optimisation directly, with others relying on the ionBench penalty function. 

### Scipy
The Scipy optimisers currently available are LM (Levenberg-Marquardt), Nelder-Mead, Powell's simplex method, Conjugate Gradient Descent, SLSQP, and Trust Region Reflective. 

Both Trust Region Reflective and LM make use of the alternative cost functions, in this case `benchmarker.signed_error()` returning a vector of residuals rather than the RMSE cost. In this case, `benchmarker.grad(residuals=True)` is used to calculate the jacobian of the residuals rather than the gradient of the RMSE cost function. 

### External Optimisers
We currently have twenty-one optimisers defined from other papers for ion channel fitting.

## Modifications
Some optimisation algorithms come with recommendations for other characteristics such as log-transforms or bounds to be applied to the parameters. These are implemented in ***ionBench*** through the use of modifications. Modifications store benchmarker settings, such as to log-transform parameters, add bounds, or use scale factors, that can then be applied to any benchmarker problem object. 

They can be particularly useful if you want to apply problem-specific settings, such as bounds determined by the problem-specific sampling function. In addition to constructing a new modification, standard modifications defined by different papers can also be loaded.

Each optimiser has a `.get_modification()` function which will get a modification relevant to that particular optimiser. The choice of modification can be changed by varying the optional input `modNum`.

## Uncertainty
There are uncertainty and unindentifiability tools built into ***ionBench***. We provide a profile likelihood calculator, which will generate, plot and save profile likelihood plots for the inputted benchmarker problem.

# Utility functions
## Summary
This directory contains a range of useful function that are used throughout the ***ionBench*** codebase.

## Autodiff
To calculate the gradient of the cost function, we require solving the model with sensitivities. These sensitivities track the derivative of the output over time with respect to each of the parameters. Since we begin all model solves from the steady state, the sensitivities are non-zero initially, so we are required to find the sensitivities of these initial conditions.

We do this through the automatic differentiation package MyGrad. This allows us to run automatic differentiation on myokit internal functions, such as the initial condition functions. In the `Benchmarker.set_steady_state()` function, if the model is a Hodgkin Huxley model, it performs automatic differentiation through the internal `_steady_state_function()` function. If the model is a Markov model instead, it no internal myokit functions satisfy the requirements for MyGrad.

The file [autodiff.py](../../ionbench/utils/autodiff.py) contains an edited copy of the Markov model steady state calculation that uses MyGrad instead of numpy. 

Checks in the `set_steady_state()` function ensures that the steady state calculations are accurate to the current myokit implementation. If they are different, an error is thrown, and if myokit detects an issue (like an unstable steady state), then the simulation uses fixed initial states instead (those defined in the *.mmt* files) with all ion channels fully closed.

This file also contains a linear system solver (`linalg_solve()`)that allows MyGrad autodiff to be used.

## Cache
This [file](../../ionbench/utils/cache.py) contains functions to cache model solves. This isn't required, and the tracker will ignore repeated parameters, but it can speed up some optimisers, at an increased memory cost.

The three functions that can be cached are:
* `cost`
* `grad`
* `signed_error`

Cached memory can be cleared with `ionbench.utils.cache.clear_all_caches()`.

The caching can be disabled by setting `ionbench.cache_enabled = False`, in which case these functions can still be used, but will not cache any results.

## Pints Classes
The [pints classes](../../ionbench/utils/classes_pints.py) file acts as a simple interface between ***pints*** and ***ionBench***. The `pints_setup` prepares a ***pints*** optimisation controller from a benchmarker with ***pints*** boundaries appropriately loaded. 

It also is able to load an advanced boundaries object to handle the rate bounds, and the `pints_setup` function can include this.

## Scipy Setup
Similar functionality is provided for ***scipy*** optimisers in the [scipy_setup](../../ionbench/utils/scipy_setup.py) file. 

The `minimize_bounds` function prepares a ***scipy*** optimisation bounds object from the parameter bounds (not the rate bounds). 

We also have a `least_squares()` function that calls `scipy.optimize.least_squares()` with the appropriate bounds, gradient function with residuals and any additional keyword arguments.

## Multistart
The [multistart](../../ionbench/utils/multistart.py) file contains the `multistart()` function that can be used to automatically run multiple optimisations from different starting points. 

A list of parameters can be passed to the function as `initParams`, and multistart will begin an optimisation from each of them, saving the tracker for each under an inputted filename.

Modifications need to be applied to the benchmarker beforehand.

It takes inputs of an optimiser's `run()` function, and additional keyword arguments to pass to the optimiser (like hyperparameters for an approach).

## Particle Optimisers
The [particle_optimisers](../../ionbench/utils/particle_optimisers.py) file contains the Particle class which is stores a parameter vector/position, a velocity, a current cost and the best cost and position. It also defines all the defaults for ***ionBench*** particle swarm optimisers.

The particle positions are stored in [0,1] space, which is mapped to [lb,ub] (so long as the approach has parameter bounds). This is handled by the `transform()` and `untransform()` methods.

The initial position is set, using `set_position()`, to +-10% of a starting location, and then clamped to the bounds.

The initial velocity is uniformly distributed between 0 and 0.1 in each parameter direction. 

The `set_cost()` function calculates the cost at the current position, and updates the best cost and position if it is lower. It also allows you to specify the cost to skip its calculation (for example, when the cost and position are set together during the hybrid PSO/TRR from Loewe et al. 2016).

Various changes are made to override these methods in the optimisers themselves.

## Population Optimisers
The [population_optimisers](../../ionbench/utils/population_optimisers.py) file contains some functions to run population-based optimisers, like genetic algorithms. 

It contains the `Individual` class which stores a parameter vector and an associated cost. It has a method `find_cost()` that calculates and stores the cost of the individual.

The `get_pop()` function generates a list of __Individual__s from a benchmarker, centred around a given parameter vector (+-50% in each parameter, in untransformed space).

`find_pop_costs()` loops through a list of individuals and calls `find_cost()` on each.

`tournement_selection()` runs tournament selection on a list of individuals, returning the selected population. TODO: Should this run without replacement?

`one_point_crossover()` performs the SinglePointCrossover from ***pymoo*** on a list of individuals, returning the new population.

`add_pymoo()` appends a pymoo population to a list of individuals (***ionBench*** population).

`sbx_crossover()` performs simulated binary crossover from ***pymoo*** on a list of individuals, returning the new population.

`polynomial_mutation()` performs polynomial mutation from ***pymoo*** on a list of individuals, returning the new population.

`get_elites()` returns the top `n` individuals from a population.

`set_elites()` replaces the worst individuals with the inputted elites.

## Results
The [results](../../ionbench/utils/results.py) file contains the function for calculating the MUE (Median Unbiased Estimator) of the success rate. It also contains a function for calculating the ERT (Expected Run Time) until a success.

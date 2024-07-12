# Problems
## Summary
This directory contains the benchmark problems used in ***ionBench***. These problems all inherit from the main [Benchmarker class](../../ionbench/benchmarker). Most of the functionality is given by that class, with the problems defined data and variables to be used by those methods.

The models and output data are stored in the [data](../../ionbench/data) directory.

Each problem has a `generate_data()` function associated with it to generate the output data files that are stored in the [data](../../ionbench/data) directory.

## Staircase
There are two problems defined [here](../../ionbench/problems/staircase.py). There is a combined `StaircaseBenchmarker` problem class that loads the staircase voltage protocol, sets the parameter and rate bounds, compiles the myokit models into simulation objects, defines the maximum time and timestep for the data, and the sampling function.

The protocol is mostly loaded as a myokit protocol, but this does not allow ramps. To handle this, we have an `add_ramps()` method which edits the myokit models to include these ramps.

The `HH` class defines the Staircase HH problem. It inherits from the `StaircaseBenchmarker` class, which itself inherits from `Benchmarker`. This loads the myokit model and sets variables to be able to set and read data from it (like the name of the output/current variable, the container for the parameters). It also defines the true parameters, name of the problem, the rate functions, which parameters should be log transformed, the cost threshold and the solver tolerances.

A simular class exists for the `MM` problem.

The definition of these two problems come from a range of sources, but the Staircase HH problem is very similar to the work in Clerx et al. 2019.

Normally distributed noise is added to the generated data. The standard deviation of this noise is 5% of the mean value of the absolute value of the current.

## Loewe
The Loewe problems are defined similarly to the Staircase ones. A `LoeweBenchmarker` class is defined that inherits from the main `Benchmarker` class. The `IKr` and `IKur` problems then inherit from this class.

The protocol is defined in the `LoeweBenchmarker` class and is generated on initialisation rather than loaded from a file.

No noise is added to the generate data.

## Moreno
The Moreno `INa` problem class features some additional complexities. The protocol is generated on initialisation, and removes one of the protocols used in Moreno et al. 2016. The removed protocol was considerably longer than the other four, and so was removed to reduce the computational cost of the problem.

Due to the summary statistic calculations needed for the Moreno problem, its class overrides the `solve_model()` and `solve_with_sensitivities()` methods. These make calls to the `sum_stats()` method, which calculates the summary statistics for the problem. The `solve_with_sensitivities()` method also calculates the sensitivities of the summary statistics, using the sensitivity of the current to derive a finite difference step through the `sum_stats()` method.

The `rmse()` method is also redefined to account for applied weights on each summary statistic curve.

No noise is added to the generate data.

## Test
This is a simple test problem to check that the code is working. It is a simple problem that doesn't require solving an ODE, ensuring that it can be run quickly. It has a clear global optimum with no other local optima. The problem is defined in the [test.py](../../ionbench/problems/test.py) file.

It uses a fake myokit simulation object to ensure it remains compatible with ***ionBench***.
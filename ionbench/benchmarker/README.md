# Benchmarker Class 
## Summary
The Benchmarker class, contained in *benchmarker.py*, is one of the two core classes in ionBench (the other being the [tracker](../../ionbench/tracker)). It provides the interface between the optimiser and the internals of ionBench (tracker, models, etc.). 

This class is unlikely to be useful on its own. It is designed to be inherited by the problems, so defines common methods and attributes that are useful for all problems, which then act on problem specific data.

The class contains many methods and attributes, so we highlight just the most important case, the cost function, and refer the interested reader to the [code](../../ionbench/benchmarker/benchmarker.py) and its docstring documentation for more details.

## Evaluating model cost
The evaluation of the cost begins with the `cost()` method call. It takes a required input of the parameters, one optional input which disables some tracking (and should only be used by ionBench internals) and returns the RMSE cost.

To calculate the cost, it is required to solve the model, which begins with a `simulate()` call. This method handles all of the setup for model solving, interfacing with the tracker, parameter transforms, and parameter bound penalties.

The `simulate()` call takes required inputs of parameters and times to solve the model, and an optional input of `continueOnError` which can be used to decide if errors in model solving should be ignored. Additionally, the `incrementSolveCounter` from the `cost()` call is also passed through to `simulate()`.

The call begins by converting the model from the input parameter space (log transformed, or scale factors) to the original parameter space that is used to solve the model. It then forces myokit to reset the simulation object. 

It verifies the parameters are inside any specified bounds, and if not, adds a penalty to the cost. It then sets the parameters and initial conditions (steady state, controlled by `set_steady_state()` which uses internal myokit functions) for the model.

Finally, it calls the `solve_model()` method while timing the call to pass to the tracker class. This method is responsible for solving the model and returning the results.

The `solve_model()` call returns the current (accessed from myokit by `self._OUTPUT_NAME`). Finally, the `rmse()` method is called to calculate the RMSE cost. This `rmse()` method is overridden for the Moreno INa problem, where a weighted RMSE is used.

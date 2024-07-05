# Tracker class
## Summary
The tracker class is one of the two main classes in ionBench (the other being the [benchmarker](../../ionbench/benchmarker)). This class stores data over an optimisation run and uses it to check for convergence.

It isn't need to be accessed by a user looking to evaluate their own optimisation approach, so the methods and attributes documented here should be considered as a look into the internal workings of ionBench.

A tracker object is initialised when a benchmarker is created, or when it is reset with `bm.reset()`. The tracker stores the following information, which is updated on every `bm.simulate()` (`bm.cost()`, `bm.signed_error()`, and `bm.squared_error()`) call and on every `bm.grad()` call though calls to `update()`:
* `self.costs`: A list of the costs of every evaluated parameter vector.
* `self.costSolves`: A list of the number of cost solves (solves without sensitivities) at each evaluation.
* `self.gradSolves`: The same but for solves with sensitivities.
* `self.evals`: A list of tuples of the evaluated parameters and the solve type (cost or grad) at each evaluation.
* `self.bestCosts`: A list of the best cost seen so far at each evaluation. Equivalent to `[np.min(self.costs[:i+1]) for i in range(len(self.costs))]`.
* `self.costTimes`: A list of the time taken for cost solves at each evaluation. 0 for grad solves or out of bounds evaluations.
* `self.gradTimes`: The same but for grad solves.
* `self.bestParams`: The best parameters seen so far (lowest cost).
* `self.bestCost`: The best cost seen so far (includes cost from grad evaluations).
* `self.firstParams`: The first parameters evaluated at the start of the optimisation.
* `self.costSolveCount`: The current number of cost evaluations. Stored as an integer
* `self.gradSolveCount`: The current number of grad evaluations. Stored as an integer
* `self.maxIterFlag`: A flag to indicate that the optimisation terminated due to either eaching the maximum number of iterations or the maximum number of function evaluations.

In addition to `update()`, the tracker also has the following methods:
* `plot()`
* `save()`
* `load()`
* `report_convergence()`
* `when_converged()`
* `cost_threshold()`
* `cost_unchanged()`
* `total_solve_time()`
* `check_repeated_params()`

## Plot
The `plot()` method is called during `bm.evaluate()` (if `bm.plotter=True`). This method plots the cost, the best cost, and histograms of the times for cost and grad solves.

## Save and Load
The `save()` method pickles a dictionary containing the tracker's attributes to a file. The `load()` method can be used to read from this dictionary to restore the tracker's state. It is used to save optimisation runs to be analysed later.

## Report Convergence
The `report_convergence()` method is called during `bm.evaluate()`. It prints some summary information from the tracker, like the best cost and termination reason.

## When Converged
This method determines at which index in the tracker data (so which parameter evaluation), a convergence state (cost threshold or cost unchanged) was reached.

## Cost Threshold
This method checks if the optimisation has achieved a cost of below the specified threshold so far, or if the optional parameter index is set, if the best cost by that index was below the threshold. It is called during `bm.is_converged()`.

## Cost Unchanged
This method does a similar check to the cost threshold, but is instead looking to see if the cost has remained unchanged over 2500 evaluations (not included out of bounds evaluations) so far, or up until index if specified. It is called during `bm.is_converged()`.

## Total Solve Time
Finds the total solve time for the optimisation, up until the specified index `i`. Returns a tuple of the total cost solve time and the total grad solve time.

## Check Repeated Params
This method checks if the inputted parameters have been evaluated before in the optimisation. It verifies against how the model was solved. The parameters did not need to be solved if:
* The parameters were evaluated previously with the same solve type (cost or grad).
* The parameters were previously evaluated as a grad and are now being evaluated as a cost (cost can be returned for free from grad).

If either of these is true for the current evaluated parameters during `update()`, then the cost/grad solve count will not be incremented and the solve time will be ignored.

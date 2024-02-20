# Changelog

## v0.4.0-alpha
Planned changes:
* Improved bounds
* Refactoring benchmarker attributes
* Standardising assumptions in optimisers
* Standard termination criteria on all optimisers
* Set seeds across ionBench

## v0.3.4-alpha - Current release
Initial state setting bug remaining in the previous release. Fixed here.

## v0.3.3-alpha
Bug fix in initial state setting to allow optimisation of remaining problems.

## v0.3.2-alpha
Improvements and updates made to profile likelihood. Markov model staircase problem no longer features noise as it interferes with the identifiability.

## v0.3.1-alpha
Tracker now records and saves the times for each of the model solves, separated into cost and gradient solves.

## v0.3.0-alpha
Added many more optimisers. Introduced lists of approaches to loop through and run everything. Added more features to the Tracker (now tracks time for model solves, reports if model was solve multiple times at the same parameters, and tracks the best parameters). Added more tests for the optimisers through a new problem specifically for ensuring all optimisers can optimise a simple problem. All problems now simulate from steady state for current parameters, rather than steady state for true parameters. Gradient calculator doesn't yet account for this.

## v0.2.0-alpha
Upgraded the gradient calculator to calculate the gradient of the cost function (or jacobian of the residuals) with respect to the parameters using myokit sensitivities. Added more optimisers which make use of this, such as a new SPSA algorithm, better matching the original from Spall 1998. Also added conjugate gradient descent, simulated annealing by Vanier 1999 and curvilinear gradient descent by Dokos 2004.

## v0.1.1-alpha
Added unit tests and removed old test scripts. Approaches have been relabelled to modifications. A derivative calculator has been implemented but will still see more work before it can be used by the optimisers.

## v0.1.0-alpha
Initial release. Contains two staircase problems, two problems from Loewe et al. and a problem from Moreno et al. Includes approaches, uncertainty quantification, and optimisers from pints, scipy, spsa and a collection of 5 published algorithms

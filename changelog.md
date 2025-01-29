# Changelog

## v0.5.1-alpha - Current release
Improved documentation. Some small tweaks to approaches. New significance testing.

## v0.5.0-alpha
Refactored tracker (breaks compatibility with previous versions). Improved memory usage from caching. Added rProp optimiser from Pints. Other minor bug fixes and improvements.

## v0.4.3-alpha
More optimiser, benchmarker and tracker bug fixes and improvements. 

## v0.4.2-alpha
Fixed bugs in pints optimisers for Moreno. Corrected the stochastic search optimisers. 

## v0.4.1-alpha
Hot fix: Fixed a bug in the steady state calculation for markov problems. 

## v0.4.0-alpha
There are a lot of changes here. A brief summary of the key ones are given below:


Lots of refactoring, both in optimisers and problems. 
Added utils module for common functions.
Bug fixes in some optimisers.
Changes to rate bounds. 
Sped up the moreno problem so that it can now be used. 
Staircase protocol now uses myokit.Protocol. 
Added cost thresholds. 
Added penalty function for out of bounds. 
More information now tracked with the Tracker. 
Updated modifications, removing some unnecessary options. 
Staircase problem now has rate bounds built in fully. 
Increased frequency of simulated data. 
Fixed bugs in Loewe IKr and IKur. 
Improvements to profile likelihood calculations and plots. 
Standardised assumptions in optimisers. 
Introduced standard termination criteria across all optimisers.

## v0.3.4-alpha
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

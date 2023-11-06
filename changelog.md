# Changelog

## v0.3.0-alpha
Planned changes:
* More approaches (particularly genetic algorithms)
* Better bounds
* Tracking time for each model solve
* Refactoring benchmarker attributes

## v0.2.0-alpha - Current release
Upgraded the gradient calculator to calculate the gradient of the cost function (or jacobian of the residuals) with respect to the parameters using myokit sensitivities. Added more optimisers which make use of this, such as a new SPSA algorithm, better matching the original from Spall 1998. Also added cojugate gradient descent, simulated annealing by Vanier 1999 and curvilinear gradient descent by Dokos 2004.

## v0.1.1-alpha
Added unit tests and removed old test scripts. Approaches have been relabelled to modifications. A derivative calculator has been implemented but will still see more work before it can be used by the optimisers.

## v0.1.0-alpha
Initial release. Contains two staircase problems, two problems from Loewe et al and a problem from Moreno et al. Includes approaches, uncertainty quantification, and optimisers from pints, scipy, spsa and a collection of 5 published algorithms

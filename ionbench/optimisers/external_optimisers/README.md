# External Optimisers
External optimisers are those that we were unable to find reliable implementations for. These are optimisers that are not included in the ***scipy*** or ***pints*** packages. We have included our own implementations of the optimisers in this directory for use in ***ionBench***.

Many of these optimisers make use of the [ionbench.utils](../../../ionbench/utils) module, especially the [particle_optimisers](../../../ionbench/utils/particle_optimisers.py) and [population_optimisers](../../../ionbench/utils/population_optimisers.py) modules.

## Curvilinear Gradient Descent - Dokos2004
This optimiser is an expansion upon one published by Dokos in 2003. It searches in a curved direction starting in the steepest descent direction and ending at an estimate for the local minimum. It then optimises along this line using Brent's method (from ***scipy***). 

The update in 2004 reweights the residual vector upon convergence to aid in identifying the global optimum. 

This optimiser also implements its own transformation of the parameter space.

The reweighting applies if the cost is non-zero, as stated in Dokos and Lovell 2004. This presents issues in noisy data, where a cost of zero is not achievable. We do not resolve this issue in the ionBench implementation.

When using Brent's method, a bracketing interval (xl, x, xu) must be defined such that f(xl) > f(x) < f(xu). The values of xl and xu are clear, 0 and inf (1e9 is used for calculation’s sake) respectively. Finding a value of x that satisfies this can be challenging and guidance is not given in Dokos and Lovell 2004. We make a handful of attempts to find a suitable bracketing interval, but if we are unable to find one, we assume the optimiser has converged and terminate. We also make some other assumptions about this bracketing interval, such as jumping to the local minimum estimate is its cost is below the initial estimate for Brent's method.

This optimiser is used in the Dokos2004, Guo2010 and Abed2013 approaches (all identical).

## Hybrid Differential Evolution/Levenberg Marquardt - Zhou2009
This optimiser uses differential evolution, and then calls Levenberg Marquardt every 1000 generations on each particle. The differential evolution algorithm used by Zhou is not stated directly. Instead, a citation is given to Storn 1996 which describes multiple DE algorithms. The DE/rand/1 algorithm appears to be the one used by Zhou, so that is what is implemented in ionBench.

An initialisation routine is not given, so we use the same initialisation as the other optimisers.

Hyperparameters are not given in Zhou 2009, so we use the recommendations from Storn 1996.

The only approach that uses this optimiser is Zhou2009.

## Genetic Algorithm - Bot2012
The genetic algorithm from Bot 2012 uses tournament selection, simulated binary crossover (SBX), a polynomial mutation function and elitism. All these steps use the implementations in [utils.population_optimisers.py](../../../ionbench/utils/population_optimisers.py).

There are two approaches that use this optimiser, Bot2012 and Groenendaal2015, which differ in hyperparameters.

## Genetic Algorithm - Smirnov2020
The genetic algorithm from Smirnov 2020 is an adaption of Bot 2012. Most aspects use [utils.population_optimisers.py](../../../ionbench/utils/population_optimisers.py), however the Cauchy mutation function is defined separately. 

There is only one approach that uses this optimiser, Smirnov2020.

## Genetic Algorithm - Cairns2017
The Cairns 2017 genetic algorithm uses a different tournament selection, crossover on the bit string of parameters, a normally distributed mutation function and elitism. Most of these are implemented in the [function](../../../ionbench/optimisers/external_optimisers/GA_Cairns2017.py).

The crossover on the mutation has a fixed degree of precision, 6 decimal places. Information beyond this is lost by rounding. The original method uses parameters that are all order 1, so this is not an issue. However, in ionBench, the parameters are not all order 1, so this makes optimisation more challenging. This could be resolved by using transforms, but this is not done here.

There is only one approach that uses this optimiser, Cairns2017.

## Genetic Algorithm - Gurkiewicz2007a
The first genetic algorithm from Gurkiewicz 2007 uses tournament selection, one point crossover, a mutation function using the problems sampling function, and elitism. Most of these are implemented in [utils.population_optimisers.py](../../../ionbench/utils/population_optimisers.py), except for the mutation function. 

This optimiser is used in the Gurkiewicz2007a and BenShalom2012 approaches. TODO: Are these unique? Table says yes but can't see why in the modification. What does ionbench.APP_UNIQUE say?

## Genetic Algorithm - Gurkiewicz2007b
The second genetic algorithm from Gurkiewicz 2007 uses a normally distributed mutation method, defined in the [function](../../../ionbench/optimisers/external_optimisers/GA_Gurkiewicz2007b.py).

The only approach that uses this optimiser is Gurkiewicz2007b.

## Particle Swarm Optimisation - Loewe2016
The first external optimiser from Loewe 2016 runs the particle swarm optimisation. This optimiser uses a fairly typical PSO algorithm, with functions given in [utils.particle_optimisers.py](../../../ionbench/utils/particle_optimisers.py). It redefines the parameter clamp function.

The initial sampling procedure for the population is assumed to be the ionBench default.

This optimiser is used in the Loewe2016b approach.

## Hybrid PSO+TRR - Loewe2016
Once PSO has completed, TRR is run on the best `M=12` particles.

The initial sampling procedure for the population is assumed to be the ionBench default.

The original TRR implementation is Matlab's lsqnonlin, which is not available in Python. We use the ***scipy*** implementation instead. This implementation does not allow a maximum number of iterations to be specified, only the maximum number of function calls, which we set to be double the number of iterations in Loewe 2016 (10 function calls).

This optimiser is used in the Loewe2016c approach.

## Hybrid PSO/TRR - Loewe2016
This optimiser calls TRR on all particles, every PSO iteration. Velocities are updated based on the TRR step.

The initial sampling procedure for the population is assumed to be the ionBench default.

The original TRR implementation is Matlab's lsqnonlin, which is not available in Python. We use the ***scipy*** implementation instead. This implementation does not allow a maximum number of iterations to be specified, only the maximum number of function calls, which we set to be double the number of iterations in Loewe 2016 (10 function calls).

This optimiser is used in the Loewe2016d approach.

## Hybrid PSO/TRR+TRR - Loewe2016
This optimiser calls TRR every PSO iteration, and then again on the best `M=12` particles.

The initial sampling procedure for the population is assumed to be the ionBench default.

The original TRR implementation is Matlab's lsqnonlin, which is not available in Python. We use the ***scipy*** implementation instead. This implementation does not allow a maximum number of iterations to be specified, only the maximum number of function calls, which we set to be double the number of iterations in Loewe 2016 (10 function calls).

This optimiser is used in the Loewe2016e approach.

## Hybrid PSO+NM - Clausen2020
This optimiser runs PSO and then NM from the best parameters. The PSO algorithm is not stated. We use ***pints*** for both the PSO and NM optimisers, making use of the [utils.classes_pints](../../../ionbench/utils/classes_pints.py).

It always runs 1000 iterations of PSO, and then Nelder Mead until convergence.

This optimiser is used in the Clausen2020 approach.

## Hybrid PSO/NM - Liu2011
This optimiser runs PSO and Nelder Mead simultaneously. The PSO algorithm is not stated, so we use a standard algorithm. 

There are a handful of things lacking clarification for this optimiser, where we have have to make assumptions. For example, the velocity of demoted Nelder Mead particles, or the hyperparameter `eps` which is set to `1e-6`. 

This optimiser is only used in the Liu2011 approach.

## Pattern Search - Kohjitani2022
The pattern search algorithm from Kohjitani 2022 is a fairly standard pattern search algorithm. It searches in a + pattern, shrinking when no parameters in the + are better than the centre. 

This optimiser is used in the Kohjitani2022 approach.

## Perturbed Particle Swarm Optimisation - Chen2012
This optimiser is a PSO optimiser that applies random perturbations to the particle positions if no improvement is observed. The description of the optimiser is sufficient for a full implementation. However, the initial sampling does not allow a single parameter to be specified, so is overridden with the ionBench default.

The perturbation makes use of groups of parameters. Here, we just assume each parameter is in its own group.

This optimiser is used in the Chen2012 approach.

## Particle Swarm Optimisation - Cabo2022
Little information on the PSO algorithm is given in Cabo 2022. However, they do provide [code](https://github.com/kkentzo/pso) with a range of possible implementations. We use the defaults, also stated here.

* There is a choice of topologies for determining the global acceleration term, in which case we use the ring topology (PSO_NHOOD_RING).
* There is a choice of inertia weights, either linearly decreasing (user specified limits) or constant (0.7298). We use linearly decreasing from 0.7298 to 0.3 over maxIter iterations.
* We use the internally calculated population size formula.
* We assume the coefficients c1 and c2 are maintained at their defaults (1.496).
* We assume the particle positions are clamped to the bounds.

This optimiser is used in the Cabo2022 approach.

## Particle Swarm Optimisation - Seemann2009
There are a couple of variations of PSO implemented in Seemann 2009. We use linearly changing acceleration constants and a random inertia weight.

This is the only PSO optimiser to not use bounds on the parameters.

The initial sampling procedure is not given so ionBench defaults are used [utils.particle_optimisers.py](../../../ionbench/utils/particle_optimisers.py).

It is used in the Seemann2009a approach.

## Random Search - Vanier1999
This optimiser was used as a 'control' in Vanier 1999. It is a simple random search algorithm that samples the parameter space, keeping track of the best parameter found.

It is only used in the Vanier1999d approach.

## Simulated Annealing - Vanier1999
This simulated annealing optimiser is based on Nelder Mead. It runs Nelder Mead but comparisons for new points are made using a simulated annealing acceptance criterion. This always accepts the new point if it is better than the current point, and accepts it with a probability if it is worse. The probability is derived from a temperature parameter which decreases over time.

Vanier originally described this temperature parameter as decreasing to 0 after maxIter iterations. We instead have it decrease to 0 over 1000 iterations.

This optimiser is used in the Vanier1999b approach.

## Stochastic Search - Vanier1999
Stochastic search is a simple optimiser the moves through parameter space, proposing new parameters from a normally distributed perturbation kernel. This perturbation kernel decreases in variance over time, before resetting to its original size.

This was originally described in Foster et al. 1993. Vanier describes the variance as decreasing linearly over time, but Foster describes a geometric decrease. The hyperparameters in Vanier 1999 suggest that the linear description may be a mistake, so we use the geometric decrease.

This optimiser is used in the Vanier1999c approach.

## Simultaneous Perturbation Stochastic Approximation - Maryak1998
This optimiser was first given in Spall 1998. It is a gradient descent optimiser that first randomly samples a direction, then finds the gradient in that direction through finite differences. Then performs a step in the decreasing direction of a fixed step size. This is repeated until convergence.

This optimiser was originally designed to minimise the number of cost function calls needed to find the gradient by using a random direction. To avoid needing to implement a finite difference calculation into ionBench (which can be challenging due to solver tolerances and noise), we use the ionBench gradient calculation instead. The final number of cost function evaluations for this optimiser is then given as twice the number of gradient evaluations, which is applied adhoc in the results, which is what it would be if a finite difference gradient calculator had been used.

This optimiser is used in the Maryak1998 approach.

#TODO: Rename spall to maryak

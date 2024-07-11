# Scipy Optimisers
In ionBench, we have implementations of some optimisers from ***scipy***. Some of these optimisers use ionbench.utils.scipy_setup to simplify the formatting for ***scipy***.

A summary of ionbench.utils.scipy_setup is given in the corresponding [README.md](../../../ionbench/utils) file.

There may be differences between the implementations of the optimisers used in the original papers (for example, Matlab implementations) compared with the scipy implementations. Most notably, in the SLSQP implementation (where the original code has been lost).

## Conjugate Gradient Descent
The Conjugate Gradient Descent method uses ***scipy***'s `optimize.minimize` function with `method='CG'`. It sets the hyperparameter gtol to 0.001 for all approaches. It is used in the Vanier1999a and Sachse2003a approaches. 

## Levenberg Marquardt
The Levenberg Marquardt method uses ***scipy***'s `optimize.least_squares` function with `method='lm'`. This uses ionbench.utils.scipy_setup to format the problem. It is used in the Balser1990a and Clancy1999 approaches (both identical).

## Nelder Mead
The Nelder Mead optimiser uses ***scipy***'s `optimize.minimize` function with `method='Nelder-Mead'` with `xtol=1e-4` and `ftol=1e-4`. It is used in the Balser1990b, Davies2012, and Moreno2016 approaches.

## Powell
This optimiser uses Powell's method, implemented in ***scipy***'s `optimize.minimize` function with `method='powell'` with `xtol=1e-4` and `ftol=1e-4`. It is used in the Sachse2003b, Seemann2009b, and Wilhelms2012a approaches (all identical).

## SLSQP
The Sequential Least Squares Quadratic Programming (SLSQP) method uses ***scipy***'s `optimize.minimize` function with `method='SLSQP'`. Rate bounds are applied through equality constraints on the penalty function. It is used in the BuenoOrovio2008 approach.

## Trust Region Reflective
The Trust Region Reflective method uses ***scipy***'s `optimize.least_squares` function with `method='trf'`. This uses ionbench.utils.scipy_setup to format the problem. It is used in the Wilhelms2012b, Du2014, and Loewe2016a approaches (all of which are identical).

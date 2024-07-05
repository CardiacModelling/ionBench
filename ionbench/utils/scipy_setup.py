import ionbench
import scipy
import numpy as np


def least_squares(bm, x0, debug, method, maxIter, **kwargs):
    """
    Run a scipy least_squares optimiser. Will use cached signed_error and jacobians and return the scipy optimisation output.
    Parameters
    ----------
    bm : Benchmarker
        A benchmarker object.
    x0 : list
        Initial parameter vector from which to start optimisation. If None, a randomly sampled parameter vector is retrieved from bm.sample().
    debug : bool
        If True, prints out the cost and parameters found by the algorithm.
    method : str
        The method to pass into least_squares.
    maxIter : int
        Maximum number of iterations.
    kwargs : dict, optional
        Additional keyword arguments to pass to least_squares.

    Returns
    -------
    out : scipy.optimize.OptimizeResult
        The result of the optimisation.
    """
    def grad(p):
        return bm.grad(p, residuals=True)

    if x0 is None:
        x0 = bm.sample()
        if debug:
            print('Sampling x0')
            print(x0)

    if debug:
        verbose = 2
    else:
        verbose = 1
    if bm.parametersBounded and method != 'lm':
        bounds = minimize_bounds(bm)
    else:
        bounds = (-np.inf, np.inf)
    out = scipy.optimize.least_squares(bm.signed_error, x0, method=method, bounds=bounds, jac=grad, verbose=verbose, max_nfev=maxIter,
                                       **kwargs)

    if out.nfev >= maxIter:
        bm.set_max_iter_flag()

    if debug:
        print(f'Cost of {out.cost} found at:')
        print(out.x)

    return out


def minimize_bounds(bm):
    """
    Map the benchmarker bounds to the scipy bounds format.
    Parameters
    ----------
    bm : Benchmarker
        A benchmarker object.
    Returns
    -------
    bounds : scipy.optimize.Bounds
        Scipy bounds object which restricts scipy optimisation to remain inside parameter bounds.
    """
    lb = bm.input_parameter_space(bm.lb)
    ub = bm.input_parameter_space(bm.ub)
    bounds = scipy.optimize.Bounds(lb=lb, ub=ub, keep_feasible=True)
    return bounds

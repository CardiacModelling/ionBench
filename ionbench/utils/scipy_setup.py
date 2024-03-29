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
    signed_error = ionbench.utils.cache.get_cached_signed_error(bm)
    grad = ionbench.utils.cache.get_cached_grad(bm, residuals=True)

    if x0 is None:
        x0 = bm.sample()
        if debug:
            print('Sampling x0')
            print(x0)

    if debug:
        verbose = 2
    else:
        verbose = 1

    out = scipy.optimize.least_squares(signed_error, x0, method=method, jac=grad, verbose=verbose, max_nfev=maxIter,
                                       **kwargs)

    if debug:
        print(f'Cost of {out.cost} found at:')
        print(out.x)

    return out


def minimize_bounds(bm, debug):
    """
    Map the benchmarker bounds to the scipy bounds format.
    Parameters
    ----------
    bm : Benchmarker
        A benchmarker object.
    debug : bool
        If True, prints out the bounds before and after mapping.
    Returns
    -------
    bounds : list
        A list of tuples containing the lower and upper bounds for each parameter. None indicates no bound.
    """
    lb = bm.input_parameter_space(bm.lb)
    ub = bm.input_parameter_space(bm.ub)
    bounds = []
    for i in range(bm.n_parameters()):
        if lb[i] == np.inf:
            lb[i] = None
        if ub[i] == np.inf:
            ub[i] = None
        bounds.append((lb[i], ub[i]))
    if debug:
        print('Bounds transformed')
        print('Old Bounds:')
        print(bm.lb)
        print(bm.ub)
        print('New bounds')
        print(bounds)
    return bounds

import ionBench.problems.staircase
import scipy.optimize
import numpy as np

def run(bm, x0, diff_step = 1e-3, bounds = (), maxfev = 20000):
    """
    Runs Trust Region Reflective optimiser from Scipy.

    Parameters
    ----------
    bm : Benchmarker
        A benchmarker to evaluate the performance of the optimisation algorithm.
    x0 : list
        Initial parameter vector from which to start optimisation.
    diff_step : float, optional
        Step size for finite difference calculation. The default is 1e-3.
    bounds : tuple, optional
        A tuple of two lists, a lower bound list and an upper bound list. The default is ().
    maxfev : int, optional
        Maximum number of cost function evaluations. The default is 20000.

    Returns
    -------
    xbest : list
        The best parameters identified by Trust Region Reflective.

    """
    
    if bounds == []:
        out = scipy.optimize.least_squares(bm.signedError, x0, method='trf', diff_step=diff_step, verbose=2, max_nfev = maxfev)
    else:
        out = scipy.optimize.least_squares(bm.signedError, x0, method='trf', diff_step=diff_step, verbose=2, max_nfev = maxfev, bounds = bounds)
    
    bm.evaluate(out.x)
    return out.x

if __name__ == '__main__':
    bm = ionBench.problems.staircase.HH_Benchmarker()
    bounds = ([0]*bm.n_parameters(),[np.inf]*bm.n_parameters())
    run(bm = bm, x0 = bm.sample(), bounds = bounds)
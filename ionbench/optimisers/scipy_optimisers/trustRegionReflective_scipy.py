import ionbench.problems.staircase
import scipy.optimize

def run(bm, x0 = [], diff_step = 1e-3, maxfev = 20000):
    """
    Runs Trust Region Reflective optimiser from Scipy.

    Parameters
    ----------
    bm : Benchmarker
        A benchmarker to evaluate the performance of the optimisation algorithm.
    x0 : list, optional
        Initial parameter vector from which to start optimisation. Default is [], in which case a randomly sampled parameter vector is retrieved from bm.sample().
    diff_step : float, optional
        Step size for finite difference calculation. The default is 1e-3.
    maxfev : int, optional
        Maximum number of cost function evaluations. The default is 20000.

    Returns
    -------
    xbest : list
        The best parameters identified by Trust Region Reflective.

    """
    if x0 == []:
        x0 = bm.sample()
    
    if bm._bounded:
        bounds = (bm.lb,bm.ub)
        out = scipy.optimize.least_squares(bm.signed_error, x0, method='trf', diff_step=diff_step, verbose=2, max_nfev = maxfev, bounds = bounds)
    else:
        out = scipy.optimize.least_squares(bm.signed_error, x0, method='trf', diff_step=diff_step, verbose=2, max_nfev = maxfev)
    
    bm.evaluate(out.x)
    return out.x

if __name__ == '__main__':
    bm = ionbench.problems.staircase.HH_Benchmarker()
    run(bm)

def get_approach():
    """
    No approach specified. Will use an empty approach

    Returns
    -------
    app : approach
        Empty approach

    """
    app = ionbench.approach.Empty(name = 'trustRegionReflective_scipy')
    return app

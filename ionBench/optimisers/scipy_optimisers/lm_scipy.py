import ionBench.problems.staircase
import scipy.optimize

def run(bm, x0, diff_step = 1e-3, maxfev = 20000):
    """
    Runs lm (Levenberg-Marquardt) least squares optimiser from Scipy.

    Parameters
    ----------
    bm : Benchmarker
        A benchmarker to evaluate the performance of the optimisation algorithm.
    x0 : list
        Initial parameter vector from which to start optimisation.
    diff_step : float, optional
        Step size for finite difference calculation. The default is 1e-3.
    maxfev : int, optional
        Maximum number of cost function evaluations. The default is 20000.

    Returns
    -------
    xbest : list
        The best parameters identified by LM.

    """
    out = scipy.optimize.least_squares(bm.signedError, x0, method='lm', diff_step=diff_step, verbose=1, max_nfev = maxfev)
    
    bm.evaluate(out.x)
    return out.x

if __name__ == '__main__':
    bm = ionBench.problems.staircase.HH_Benchmarker()
    run(bm = bm, x0 = bm.sample())
import ionbench.problems.staircase
import scipy.optimize

def run(bm, x0 = [], diff_step = 1e-3, maxfev = 20000):
    """
    Runs lm (Levenberg-Marquardt) least squares optimiser from Scipy.

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
        The best parameters identified by LM.

    """
    if x0 == []:
        x0 = bm.sample()
    
    out = scipy.optimize.least_squares(bm.signedError, x0, method='lm', diff_step=diff_step, verbose=1, max_nfev = maxfev)
    
    bm.evaluate(out.x)
    return out.x

if __name__ == '__main__':
    bm = ionbench.problems.staircase.HH_Benchmarker()
    run(bm)
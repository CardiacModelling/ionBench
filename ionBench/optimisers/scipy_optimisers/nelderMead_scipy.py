import ionBench.problems.staircase
import scipy.optimize
import nupmy as np

def run(bm, x0, xtol = 1e-4, ftol = 1e-4, maxiter = 5000, maxfev = 20000):
    """
    Runs Nelder-Mead optimiser from Scipy.

    Parameters
    ----------
    bm : Benchmarker
        A benchmarker to evaluate the performance of the optimisation algorithm.
    x0 : list
        Initial parameter vector from which to start optimisation.
    xtol : float, optional
        Tolerance in parameters. Used as a termination criterion. The default is 1e-4.
    ftol : float, optional
        Tolerance in cost. Used as a termination criterion. The default is 1e-4.
    maxiter : int, optional
        Maximum number of iterations of Nelder-Mead to use. The default is 5000.
    maxfev : int, optional
        Maximum number of cost function evaluations. The default is 20000.

    Returns
    -------
    xbest : list
        The best parameters identified by Nelder-Mead.

    """
    if bm._bounded:
        bounds = (bm.lb,bm.ub)
        out = scipy.optimize.minimize(bm.cost, x0, method='nelder-mead', options={'disp': True, 'xatol': xtol, 'fatol': ftol, 'maxiter': maxiter, 'maxfev': maxfev}, bounds = bounds)
    else:
        out = scipy.optimize.minimize(bm.cost, x0, method='nelder-mead', options={'disp': True, 'xatol': xtol, 'fatol': ftol, 'maxiter': maxiter, 'maxfev': maxfev})
    
    bm.evaluate(out.x)
    return out.x

if __name__ == '__main__':
    bm = ionBench.problems.staircase.HH_Benchmarker()
    bounds = [(0,None)]*bm.n_parameters()
    run(bm = bm, x0 = bm.sample(), bounds = bounds)
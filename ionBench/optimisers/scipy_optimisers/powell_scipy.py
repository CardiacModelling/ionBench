import ionBench.problems.staircase
import scipy.optimize

def run(bm, x0, xtol = 1e-4, ftol = 1e-4, maxiter = 5000, maxfev = 20000, bounds = []):
    """
    Runs Powell's Simplex optimiser from Scipy.

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
        Maximum number of iterations of Powell's Simplex to use. The default is 5000.
    maxfev : int, optional
        Maximum number of cost function evaluations. The default is 20000.
    bounds : list, optional
        Bounds to be passed into Powell's Simplex. A list of tuples containing upper and lower bounds on parameters. Use None to specify no bound. The default is [].

    Returns
    -------
    xbest : list
        The best parameters identified by Powell's Simplex.

    """
    if bounds == []:
        out = scipy.optimize.minimize(bm.cost, x0, method='powell', options={'disp': True, 'xtol': xtol, 'ftol': ftol, 'maxiter': maxiter, 'maxfev': maxfev})
    else:
        out = scipy.optimize.minimize(bm.cost, x0, method='powell', options={'disp': True, 'xtol': xtol, 'ftol': ftol, 'maxiter': maxiter, 'maxfev': maxfev}, bounds = bounds)
    
    bm.evaluate(out.x)
    return out.x

if __name__ == '__main__':
    bm = ionBench.problems.staircase.HH_Benchmarker()
    bounds = [(0,None)]*bm.n_parameters()
    run(bm = bm, x0 = bm.defaultParams, bounds = bounds)
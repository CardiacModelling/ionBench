import ionbench.problems.staircase
import scipy.optimize
import numpy as np

def run(bm, x0 = [], xtol = 1e-4, ftol = 1e-4, maxiter = 5000, maxfev = 20000):
    """
    Runs Nelder-Mead optimiser from Scipy.

    Parameters
    ----------
    bm : Benchmarker
        A benchmarker to evaluate the performance of the optimisation algorithm.
    x0 : list, optional
        Initial parameter vector from which to start optimisation. Default is [], in which case a randomly sampled parameter vector is retrieved from bm.sample().
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
    if x0 == []:
        x0 = bm.sample()
    
    if bm._bounded:
        lb = bm.lb[:] #Generate copy
        ub = bm.ub[:] #Generate copy
        for i in range(bm.n_parameters()):
            if lb[i] == np.inf:
                lb[i] = None
            if ub[i] == np.inf:
                ub[i] = None
        bounds = (lb,ub)
        
        out = scipy.optimize.minimize(bm.cost, x0, method='nelder-mead', options={'disp': True, 'xatol': xtol, 'fatol': ftol, 'maxiter': maxiter, 'maxfev': maxfev}, bounds = bounds)
    else:
        out = scipy.optimize.minimize(bm.cost, x0, method='nelder-mead', options={'disp': True, 'xatol': xtol, 'fatol': ftol, 'maxiter': maxiter, 'maxfev': maxfev})
    
    bm.evaluate(out.x)
    return out.x

if __name__ == '__main__':
    bm = ionbench.problems.staircase.HH_Benchmarker()
    run(bm)
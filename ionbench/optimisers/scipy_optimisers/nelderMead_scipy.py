import ionbench.problems.staircase
import scipy.optimize
from ionbench.utils.scipy_setup import minimize_bounds


# noinspection PyShadowingNames
def run(bm, x0=None, xtol=1e-4, ftol=1e-4, maxIter=1000, maxfev=20000, debug=False):
    """
    Runs Nelder-Mead optimiser from Scipy. Bounds are automatically loaded from the benchmarker if present.

    Parameters
    ----------
    bm : Benchmarker
        A benchmarker to evaluate the performance of the optimisation algorithm.
    x0 : list, optional
        Initial parameter vector from which to start optimisation. Default is None, in which case a randomly sampled parameter vector is retrieved from bm.sample().
    xtol : float, optional
        Tolerance in parameters. Used as a termination criterion. The default is 1e-4.
    ftol : float, optional
        Tolerance in cost. Used as a termination criterion. The default is 1e-4.
    maxIter : int, optional
        Maximum number of iterations of Nelder-Mead to use. The default is 1000.
    maxfev : int, optional
        Maximum number of cost function evaluations. The default is 20000.
    debug : bool, optional
        If True, prints out the cost and parameters found by the algorithm. The default is False.

    Returns
    -------
    xbest : list
        The best parameters identified by Nelder-Mead.
    """
    if x0 is None:
        x0 = bm.sample()
        if debug:
            print('Sampling x0')
            print(x0)

    if bm.parametersBounded:
        bounds = minimize_bounds(bm)
        out = scipy.optimize.minimize(bm.cost, x0, method='nelder-mead', options={'disp': debug, 'xatol': xtol, 'fatol': ftol, 'maxiter': maxIter, 'maxfev': maxfev}, bounds=bounds)
    else:
        out = scipy.optimize.minimize(bm.cost, x0, method='nelder-mead', options={'disp': debug, 'xatol': xtol, 'fatol': ftol, 'maxiter': maxIter, 'maxfev': maxfev})

    if out.nit >= maxIter or out.nfev >= maxfev:
        bm.set_max_iter_flag()

    if debug:
        print(f'Cost of {out.fun} found at:')
        print(out.x)

    bm.evaluate()
    return out.x


# noinspection PyShadowingNames
def get_modification(modNum=1):
    """
    modNum = 1 -> Balser1990
    modNum = 2 -> Davies2012
    modNum = 4 -> Moreno2016

    Returns
    -------
    mod : modification
        Modification corresponding to inputted modNum. Default is modNum = 1, so Balser1990.

    """

    if modNum == 1:
        mod = ionbench.modification.Balser1990()
    elif modNum == 2:
        mod = ionbench.modification.Davies2012()
    elif modNum == 3:
        mod = ionbench.modification.Moreno2016()
    else:
        mod = ionbench.modification.Empty(name='nelderMead_scipy')
    return mod

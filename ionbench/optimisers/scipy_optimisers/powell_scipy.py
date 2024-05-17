import ionbench.problems.staircase
import scipy.optimize
from ionbench.utils.scipy_setup import minimize_bounds


# noinspection PyShadowingNames
def run(bm, x0=None, xtol=1e-4, ftol=1e-4, maxIter=1000, maxfev=20000, debug=False):
    """
    Runs Powell's Simplex optimiser from Scipy. Bounds are automatically loaded from the benchmarker if present.

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
        Maximum number of iterations of Powell's Simplex to use. The default is 1000.
    maxfev : int, optional
        Maximum number of cost function evaluations. The default is 20000.
    debug : bool, optional
        If True, prints out the cost and parameters found by the algorithm. The default is False.

    Returns
    -------
    xbest : list
        The best parameters identified by Powell's Simplex.
    """
    if x0 is None:
        x0 = bm.sample()
        if debug:
            print('Sampling x0')
            print(x0)

    if bm.parametersBounded:
        bounds = minimize_bounds(bm)
        out = scipy.optimize.minimize(bm.cost, x0, method='powell', options={'disp': debug, 'xtol': xtol, 'ftol': ftol, 'maxiter': maxIter, 'maxfev': maxfev}, bounds=bounds)
    else:
        out = scipy.optimize.minimize(bm.cost, x0, method='powell', options={'disp': debug, 'xtol': xtol, 'ftol': ftol, 'maxiter': maxIter, 'maxfev': maxfev})

    if debug:
        print(f'Cost of {out.fun} found at:')
        print(out.x)

    bm.evaluate()
    return out.x


# noinspection PyShadowingNames
def get_modification(modNum=1):
    """
    modNum = 1 -> Sachse2003
    modNum = 2 -> Seemann2009
    modNum = 3 -> Wilhelms2012a

    Returns
    -------
    mod : modification
        Modification corresponding to inputted modNum. Default is modNum = 1, so Sachse2003.

    """

    if modNum == 1:
        mod = ionbench.modification.Sachse2003()
    elif modNum == 2:
        mod = ionbench.modification.Seemann2009()
    elif modNum == 3:
        mod = ionbench.modification.Wilhelms2012a()
    else:
        mod = ionbench.modification.Empty(name='powell_scipy')
    return mod


if __name__ == '__main__':
    bm = ionbench.problems.staircase.HH()
    mod = get_modification()
    mod.apply(bm)
    run(bm, debug=True, **mod.kwargs)

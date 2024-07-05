import ionbench.problems.staircase
import scipy.optimize
from ionbench.utils.scipy_setup import minimize_bounds


# noinspection PyShadowingNames
def run(bm, x0=None, maxIter=1000, debug=False):
    """
    Runs Sequential Least SQuares Programming optimiser from Scipy. An example of a Sequential Quadratic Programming method which uses a quasi-newton update strategy to approximate the hessian.

    Parameters
    ----------
    bm : Benchmarker
        A benchmarker to evaluate the performance of the optimisation algorithm.
    x0 : list, optional
        Initial parameter vector from which to start optimisation. Default is None, in which case a randomly sampled parameter vector is retrieved from bm.sample().
    maxIter : int, optional
        Maximum number of iterations of SLSQP to use. The default is 1000.
    debug : bool, optional
        If True, prints out the cost and parameters found by the algorithm. The default is False.

    Returns
    -------
    xbest : list
        The best parameters identified by SLSQP.
    """
    if x0 is None:
        x0 = bm.sample()
        if debug:
            print('Sampling x0')
            print(x0)
    if bm.ratesBounded:
        constraints = {'type': 'eq', 'fun': lambda p: bm.parameter_penalty(bm.original_parameter_space(p))+bm.rate_penalty(bm.original_parameter_space(p))}
    else:
        constraints = ()
    if bm.parametersBounded:
        bounds = minimize_bounds(bm)
        out = scipy.optimize.minimize(bm.cost, x0, jac=bm.grad, method='SLSQP', constraints=constraints, options={'disp': debug, 'maxiter': maxIter}, bounds=bounds)
    else:
        out = scipy.optimize.minimize(bm.cost, x0, jac=bm.grad, method='SLSQP', constraints=constraints, options={'disp': debug, 'maxiter': maxIter})

    if out.nit >= maxIter:
        bm.set_max_iter_flag()

    if debug:
        print(f'Cost of {out.fun} found at:')
        print(out.x)

    bm.evaluate()
    return out.x


# noinspection PyUnusedLocal,PyShadowingNames
def get_modification(modNum=1):
    """
    modNum = 1 -> BuenoOrovio2008

    Returns
    -------
    mod : modification
        Modification corresponding to inputted modNum. Default is modNum = 1, so BuenoOrovio2008.

    """
    mod = ionbench.modification.BuenoOrovio2008()
    return mod

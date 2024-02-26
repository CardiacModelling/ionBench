import ionbench.problems.staircase
import scipy.optimize
from functools import lru_cache


def run(bm, x0=[], gtol=0.001, maxIter=1000, debug=False):
    """
    Runs Conjugate Gradient Descent optimiser from Scipy.

    Parameters
    ----------
    bm : Benchmarker
        A benchmarker to evaluate the performance of the optimisation algorithm.
    x0 : list, optional
        Initial parameter vector from which to start optimisation. Default is [], in which case a randomly sampled parameter vector is retrieved from bm.sample().
    gtol : float, optional
        Tolerance in for the gradient. Gradient norm must be less than gtol before algorithm successfully terminates. The default is 0.001.
    maxIter : int, optional
        Maximum number of iterations of Conjugate Gradient Descent to use. The default is 1000.

    Returns
    -------
    xbest : list
        The best parameters identified by Conjugate Gradient Descent.

    """
    @lru_cache(maxsize=None)
    def grad(p):
        return bm.grad(p)

    @lru_cache(maxsize=None)
    def cost(p):
        return bm.cost(p)

    def grad_scipy(p):
        return grad(tuple(p))

    def cost_scipy(p):
        return cost(tuple(p))

    if len(x0) == 0:
        x0 = bm.sample()
        if debug:
            print('Sampling x0')
            print(x0)

    out = scipy.optimize.minimize(cost_scipy, x0, jac=grad_scipy, method='CG', options={'disp': debug, 'gtol': gtol, 'maxiter': maxIter})

    if debug:
        print(f'Cost of {out.fun} found at:')
        print(out.x)

    bm.evaluate()
    return out.x


def get_modification(modNum=1):
    """
    modNum = 1 -> Sachse2003
    modNum = 2 -> Vanier1999

    Returns
    -------
    mod : modification
        Modification corresponding to inputted modNum. Default is modNum = 1, so Sachse2003.

    """

    if modNum == 1:
        mod = ionbench.modification.Sachse2003()
    elif modNum == 2:
        mod = ionbench.modification.Vanier1999()
    else:
        mod = ionbench.modification.Empty(name='conjugateGD_scipy_scipy')
    return mod


if __name__ == '__main__':
    bm = ionbench.problems.staircase.HH(sensitivities=True)
    mod = get_modification()
    mod.apply(bm)
    run(bm, debug=True, **mod.kwargs)

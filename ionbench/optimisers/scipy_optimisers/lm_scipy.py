import ionbench.problems.staircase
import scipy.optimize
from functools import lru_cache


def run(bm, x0=None, maxIter=1000, debug=False):
    """
    Runs lm (Levenberg-Marquardt), the least squares optimiser from Scipy.

    Parameters
    ----------
    bm : Benchmarker
        A benchmarker to evaluate the performance of the optimisation algorithm.
    x0 : list, optional
        Initial parameter vector from which to start optimisation. Default is None, in which case a randomly sampled parameter vector is retrieved from bm.sample().
    maxIter : int, optional
        Maximum number of cost function evaluations. The default is 1000.
    debug : bool, optional
        If True, prints out the cost and parameters found by the algorithm. The default is False.

    Returns
    -------
    xbest : list
        The best parameters identified by LM.
    """
    @lru_cache(maxsize=None)
    def grad(p):
        """
        Find the jacobian of the residuals. This function is cached so requires the input to be hashable (for example tuple).
        """
        return bm.grad(p, residuals=True)

    @lru_cache(maxsize=None)
    def signed_error(p):
        """
        Find the jacobian of the residuals. This function is cached so requires the input to be hashable (for example tuple).
        """
        return bm.signed_error(p)

    def grad_scipy(p):
        """
        Wrapper for the cached grad function. This is required as the scipy optimiser requires a function won't supply a hashable type as input.
        """
        return grad(tuple(p))

    def signed_error_scipy(p):
        """
        Wrapper for the cached signed_error function. This is required as the scipy optimiser requires a function won't supply a hashable type as input.
        """
        return signed_error(tuple(p))

    if x0 is None:
        x0 = bm.sample()
        if debug:
            print('Sampling x0')
            print(x0)

    if debug:
        verbose = 2
    else:
        verbose = 1

    out = scipy.optimize.least_squares(signed_error_scipy, x0, method='lm', jac=grad_scipy, verbose=verbose, max_nfev=maxIter)

    if debug:
        print(f'Cost of {out.cost} found at:')
        print(out.x)

    bm.evaluate()
    return out.x


def get_modification(modNum=1):
    """
    modNum = 1 -> Balser1990
    modNum = 2 -> Clancy1999

    Returns
    -------
    mod : modification
        Modification corresponding to inputted modNum. Default is modNum = 1, so Balser1990.

    """

    if modNum == 1:
        mod = ionbench.modification.Balser1990()
    elif modNum == 2:
        mod = ionbench.modification.Clancy1999()
    else:
        mod = ionbench.modification.Empty(name='lm_scipy')
    return mod


if __name__ == '__main__':
    bm = ionbench.problems.staircase.HH(sensitivities=True)
    mod = get_modification()
    mod.apply(bm)
    run(bm, debug=True, **mod.kwargs)

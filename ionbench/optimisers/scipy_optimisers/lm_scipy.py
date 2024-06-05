import ionbench.problems.staircase
from ionbench.utils.scipy_setup import least_squares


# noinspection PyShadowingNames
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
    out = ionbench.utils.scipy_setup.least_squares(bm, x0, debug, method='lm', maxIter=maxIter)

    bm.evaluate()
    return out.x


# noinspection PyShadowingNames
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

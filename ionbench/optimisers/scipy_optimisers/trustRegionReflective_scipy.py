import ionbench.problems.staircase
from ionbench.utils.scipy_setup import least_squares


# noinspection PyShadowingNames
def run(bm, x0=None, maxIter=1000, debug=False):
    """
    Runs Trust Region Reflective optimiser from Scipy.

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
        The best parameters identified by Trust Region Reflective.
    """
    out = least_squares(bm, x0, debug, method='trf', maxIter=maxIter)

    bm.evaluate()
    return out.x


# noinspection PyShadowingNames
def get_modification(modNum=1):
    """
    modNum = 1 -> Wilhelms2012b
    modNum = 2 -> Du2014
    modNum = 3 -> Loewe2016

    Returns
    -------
    mod : modification
        Modification corresponding to inputted modNum. Default is modNum = 1, so Wilhelms2012b.

    """

    if modNum == 1:
        mod = ionbench.modification.Wilhelms2012b()
    elif modNum == 2:
        mod = ionbench.modification.Du2014()
    elif modNum == 3:
        mod = ionbench.modification.Loewe2016()
    else:
        mod = ionbench.modification.Empty(name='trr_scipy')
    return mod

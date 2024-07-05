import pints
import ionbench
from ionbench.utils import classes_pints


# noinspection PyShadowingNames
def run(bm, x0=None, maxIter=1000, debug=False):
    """
    Runs Nelder Mead from Pints using a benchmarker.

    Parameters
    ----------
    bm : Benchmarker
        A benchmarker to evaluate the performance of the optimisation algorithm.
    x0 : list, optional
        Initial parameter vector from which to start optimisation. Default is None, in which case a randomly sampled parameter vector is retrieved from bm.sample().
    maxIter : int, optional
        Number of iterations of Nelder Mead to run. The default is 1000.
    debug : bool, optional
        If True, logging messages are printed every iteration. Otherwise, the default of every iteration for the first 3 and then every 20 iterations. The default is False.

    Returns
    -------
    xbest : list
        The best parameters identified by Nelder-Mead.

    """
    model, opt = classes_pints.pints_setup(bm, x0, pints.NelderMead, maxIter, debug, forceUnbounded=True)
    x, f = opt.run()
    if opt.iterations() >= maxIter:
        model.bm.set_max_iter_flag()
    model.bm.evaluate()
    return x


# noinspection PyUnusedLocal,PyShadowingNames
def get_modification(modNum=1):
    """
    No modification for this optimiser. Will use an empty modification.

    Returns
    -------
    mod : modification
        Empty modification

    """
    mod = ionbench.modification.Empty(name='nelderMead_pints')
    return mod

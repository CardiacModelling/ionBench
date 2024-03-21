import pints
import ionbench
import numpy as np
from ionbench.optimisers.pints_optimisers import classes_pints


# noinspection PyShadowingNames
def run(bm, x0=None, maxIter=1000, debug=False):
    """
    Runs XNES (Exponential Natural Evolution Strategy) from Pints using a benchmarker.

    Parameters
    ----------
    bm : Benchmarker
        A benchmarker to evaluate the performance of the optimisation algorithm.
    x0 : list, optional
        Initial parameter vector from which to start optimisation. Default is None, in which case a randomly sampled parameter vector is retrieved from bm.sample().
    maxIter : int, optional
        Number of iterations of XNES to run. The default is 1000.
    debug : bool, optional
        If True, logging messages are printed every iteration. Otherwise, the default of every iteration for the first 3 and then every 20 iterations. The default is False.

    Returns
    -------
    xbest : list
        The best parameters identified by XNES.

    """
    model, opt = classes_pints.pints_setup(bm, x0, pints.XNES)
    opt.set_max_iterations(maxIter)
    if debug:
        opt.set_log_interval(iters=1)
    # Run the optimisation
    x, f = opt.run()

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
    mod = ionbench.modification.Empty(name='xnes_pints')
    return mod


if __name__ == '__main__':
    bm = ionbench.problems.staircase.HH()
    mod = get_modification()
    mod.apply(bm)
    run(bm, debug=True, **mod.kwargs)

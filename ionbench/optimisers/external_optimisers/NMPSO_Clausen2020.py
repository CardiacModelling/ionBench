import pints
import ionbench
import numpy as np
from ionbench.optimisers.pints_optimisers import classes_pints
from functools import lru_cache
# Limited information on the implementation given in Clausen 2020.


def run(bm, x0=None, maxIter=1000, debug=False):
    """
    Runs PSO (Particle Swarm Optimisation) from Pints and then Nelder Mead from Pints.

    Parameters
    ----------
    bm : Benchmarker
        A benchmarker to evaluate the performance of the optimisation algorithm.
    x0 : list, optional
        Initial parameter vector from which to start optimisation. Default is None, in which case a randomly sampled parameter vector is retrieved from bm.sample().
    maxIter : int, optional
        Number of iterations of PSO and Nelder Mead to run. Maximum total number of iterations is 2*maxIter. The default is 1000.
    debug : bool, optional
        If True, debug information will be printed, reporting more optimisation information. The default is False.

    Returns
    -------
    xbest : list
        The best parameters identified by PSO.

    """

    model, opt = classes_pints.pints_setup(bm, x0, pints.PSO)
    opt.set_max_iterations(maxIter)
    if debug:
        print('Beginning PSO')
    # Run the optimisation
    x, f = opt.run()
    if debug:
        print(f'PSO complete with best cost of {f}')
    # Create an optimisation controller
    model, opt = classes_pints.pints_setup(bm, x, pints.NelderMead, forceUnbounded=True)
    opt.set_max_iterations(maxIter)
    # Run the optimisation
    if debug:
        print('Beginning NM')
    x, f = opt.run()
    if debug:
        print(f'Nelder Mead complete with best cost of {f}')
    model.bm.evaluate()
    return x


# noinspection PyUnusedLocal
def get_modification(modNum=1):
    """
    modNum = 1 -> Clausen2020

    Returns
    -------
    mod : modification
        Modification corresponding to inputted modNum. Default is modNum = 1, so Clausen2020.

    """
    mod = ionbench.modification.Clausen2020()
    return mod


if __name__ == '__main__':
    bm = ionbench.problems.staircase.HH()
    mod = get_modification()
    mod.apply(bm)
    run(bm, maxIter=20, debug=True, **mod.kwargs)

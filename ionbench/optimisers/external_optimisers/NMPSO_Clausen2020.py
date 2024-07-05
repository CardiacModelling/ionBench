import pints
import ionbench
from ionbench.utils import classes_pints


# Limited information on the implementation given in Clausen 2020.


# noinspection PyShadowingNames
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
        Number of iterations of Nelder Mead to run after PSO. PSO is hardcoded to use a maximum of 1000 iterations. The maxIter flag is set using the maximum number of iterations of Nelder Mead. The default is 1000.
    debug : bool, optional
        If True, debug information will be printed, reporting more optimisation information. The default is False.

    Returns
    -------
    xbest : list
        The best parameters identified by PSO.

    """

    model, opt = classes_pints.pints_setup(bm, x0, pints.PSO, maxIter, debug)
    opt.set_max_iterations(1000)
    if debug:
        print('Beginning PSO')
    # Run the optimisation
    x, f = opt.run()
    if debug:
        print(f'PSO complete with best cost of {f}')
    # Create an optimisation controller
    model, opt = classes_pints.pints_setup(bm, x, pints.NelderMead, maxIter, debug, forceUnbounded=True)
    # Run the optimisation
    if debug:
        print('Beginning NM')
    x, f = opt.run()
    if opt.iterations() >= maxIter:
        model.bm.set_max_iter_flag()
    if debug:
        print(f'Nelder Mead complete with best cost of {f}')
    model.bm.evaluate()
    return x


# noinspection PyUnusedLocal,PyShadowingNames
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

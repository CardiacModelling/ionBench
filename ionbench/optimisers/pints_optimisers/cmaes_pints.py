import pints
import ionbench
from ionbench.utils import classes_pints


# noinspection PyShadowingNames
def run(bm, x0=None, popSize=None, maxIter=1000, debug=False):
    """
    Runs CMA-ES from Pints using a benchmarker.

    Parameters
    ----------
    bm : Benchmarker
        A benchmarker to evaluate the performance of the optimisation algorithm.
    x0 : list, optional
        Initial parameter vector from which to start optimisation. Default is None, in which case a randomly sampled parameter vector is retrieved from bm.sample().
    popSize : int, optional
        The population size to use in CMA-ES.
    maxIter : int, optional
        Number of iterations of CMA-ES to run. The default is 1000.
    debug : bool, optional
        If True, logging messages are printed every iteration. Otherwise, the default of every iteration for the first 3 and then every 20 iterations. The default is False.

    Returns
    -------
    xbest : list
        The best parameters identified by CMA-ES.

    """
    model, opt = classes_pints.pints_setup(bm, x0, pints.CMAES, maxIter, debug)
    if popSize is not None:
        opt.optimiser().set_population_size(popSize)
    x, f = opt.run()
    if opt.iterations() >= maxIter:
        model.bm.set_max_iter_flag()
    model.bm.evaluate()
    return x


# noinspection PyShadowingNames
def get_modification(modNum=1):
    """
    modNum = 1 -> Beattie2018
    modNum = 2 -> JedrzejewskiSzmek2018
    modNum = 3 -> Clerx2019

    Returns
    -------
    mod : modification
        Modification corresponding to inputted modNum. Default is modNum = 1, so Beattie2018.

    """

    if modNum == 1:
        mod = ionbench.modification.Beattie2018()
    elif modNum == 2:
        mod = ionbench.modification.JedrzejewskiSzmek2018()
    elif modNum == 3:
        mod = ionbench.modification.Clerx2019()
    else:
        mod = ionbench.modification.Empty(name='cmaes_pints')
    return mod

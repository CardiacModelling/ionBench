import numpy as np
import ionbench


# noinspection PyShadowingNames
def run(bm, x0=None, maxIter=1000, debug=False):
    """
    Random search from Vanier et al. 1999. This optimiser randomly samples maxIter parameters and reports the best one.

    Parameters
    ----------
    bm : Benchmarker
        A benchmarker to evaluate the performance of the optimisation algorithm.
    x0 : list, optional
        Initial parameter guess. If x0=None (the default), then the population will be sampled from the benchmarker problems .sample() method.
    maxIter : int, optional
        Maximum number of iterations. The default is 1000.
    debug : bool, optional
        If True, debug information will be printed, reporting that status of the optimisation each generation. The default is False.

    Returns
    -------
    xbest : list
        The best parameters found.

    """
    cost_func = ionbench.utils.cache.get_cached_cost(bm)

    if x0 is None:
        # sample initial point
        x0 = bm.sample()
    x_best = x0
    cost_best = cost_func(x0)
    if debug:
        print(f'Starting cost is {cost_best}')
    i = None
    for i in range(maxIter):
        x_new = bm.sample()
        cost_new = cost_func(x_new)
        if debug:
            print(f'New point has cost of {cost_new}')
        if cost_new < cost_best:
            cost_best = cost_new
            x_best = x_new
            if debug:
                print('Improvement found')
        if bm.is_converged():
            break

    if i >= maxIter-1:
        bm.set_max_iter_flag()

    if debug:
        print('Complete')
        print(f'Final cost is {cost_best}')

    bm.evaluate()
    return x_best


# noinspection PyUnusedLocal,PyShadowingNames
def get_modification(modNum=1):
    """
    modNum = 1 -> Vanier1999

    Returns
    -------
    mod : modification
        Modification corresponding to inputted modNum. Default is modNum = 1, so Vanier1999.

    """
    mod = ionbench.modification.Vanier1999()
    return mod

import numpy as np
import ionbench


def run(bm, x0=[], maxIter=1000, debug=False):
    """
    Random search from Vanier et al 1999. This optimiser random sample maxIter parameters and reports the best one.

    Parameters
    ----------
    bm : Benchmarker
        A benchmarker to evaluate the performance of the optimisation algorithm.
    x0 : list, optional
        Initial parameter guess.
    maxIter : int, optional
        Maximum number of iterations. The default is 1000.
    debug : bool, optional
        If True, debug information will be printed, reporting that status of the optimisation each generation. The default is False.

    Returns
    -------
    xbest : list
        The best parameters found.

    """

    if len(x0) == 0:
        # sample initial point
        x0 = bm.sample()
    x_best = x0
    cost_best = bm.cost(x0)
    if debug:
        print(f'Starting cost is {cost_best}')
    for i in range(maxIter):
        x_new = bm.sample()
        cost_new = bm.cost(x_new)
        if debug:
            print(f'New point has cost of {cost_new}')
        if cost_new < cost_best:
            cost_best = cost_new
            x_best = x_new
            if debug:
                print('Improvement found')
    if debug:
        print('Complete')
        print(f'Final cost is {cost_best}')

    bm.evaluate(x_best)
    return x_best


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


if __name__ == '__main__':
    bm = ionbench.problems.staircase.HH()
    mod = get_modification()
    mod.apply(bm)
    run(bm, maxIter=100, debug=True, **mod.kwargs)

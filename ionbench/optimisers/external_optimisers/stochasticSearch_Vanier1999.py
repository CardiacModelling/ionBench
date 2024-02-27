import numpy as np
import ionbench


def run(bm, x0=None, varInit=0.5, varMin=0.05, varCont=0.95, maxIter=1000, debug=False):
    """
    Stochastic search from Vanier et al. 1999.

    Parameters
    ----------
    bm : Benchmarker
        A benchmarker to evaluate the performance of the optimisation algorithm.
    x0 : list, optional
        Initial parameter guess. If x0=None (the default), then the population will be sampled from the benchmarker problems .sample() method.
    varInit : float, optional
        The initial variance. The default is 0.5.
    varMin : float, optional
        The minimum variance. Once the variance is decreased below this minimum, it is reset to its initial value of varInit. The default is 0.05.
    varCont : float, optional
        The variance contraction rate. Every iteration the current variance is varCont times the previous. The default is 0.95.
    maxIter : int, optional
        Maximum number of iterations. The default is 1000.
    debug : bool, optional
        If True, debug information will be printed, reporting that status of the optimisation each generation. The default is False.

    Returns
    -------
    xbest : list
        The best parameters found.

    """

    if x0 is None:
        # sample initial point
        x0 = bm.sample()
    var = varInit
    x0_cost = bm.cost(x0)
    if debug:
        print(f'Starting cost is {x0_cost}')
    for i in range(maxIter):

        trial = x0 + np.random.normal(loc=0, scale=np.sqrt(x0 * var))
        trial_cost = bm.cost(trial)
        if debug:
            print(f'var: {var}, trial cost: {trial_cost}')
        if trial_cost < x0_cost:
            x0 = trial
            x0_cost = trial_cost
            if debug:
                print('Found improvement')
        var *= varCont
        if var < varMin:
            var = varInit
        if bm.is_converged():
            break

    if debug:
        print('Complete')
        print(f'Final cost is {x0_cost}')

    bm.evaluate()
    return x0


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

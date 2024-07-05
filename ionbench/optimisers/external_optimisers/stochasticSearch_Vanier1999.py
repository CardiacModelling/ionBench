"""
The module provides the stochastic search algorithm from Vanier et al. 1999.
This algorithm was previously described in Foster et al. 1993.
We use the hyperparameters given by Vanier et al. 1999.
Note that while Vanier et al. 1999 describes the variance as decreasing linearly, we use a geometric decrease as described in Foster et al. 1993.
"""
import numpy as np
import ionbench


# noinspection PyShadowingNames
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
    cost_func = ionbench.utils.cache.get_cached_cost(bm)

    if x0 is None:
        # sample initial point
        x0 = bm.sample()
    var = varInit
    x0_cost = cost_func(x0)
    if debug:
        print(f'Starting cost is {x0_cost}')
    i = None
    for i in range(maxIter):
        trial = x0 + np.random.normal(loc=np.zeros(bm.n_parameters()), scale=np.sqrt(var))*(bm.input_parameter_space(bm.ub)-bm.input_parameter_space(bm.lb))
        trial = bm.clamp_parameters(trial)
        trial_cost = cost_func(trial)
        if debug:
            print(x0)
            print(trial)
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

    if i >= maxIter-1:
        bm.set_max_iter_flag()

    if debug:
        print('Complete')
        print(f'Final cost is {x0_cost}')

    bm.evaluate()
    return x0


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

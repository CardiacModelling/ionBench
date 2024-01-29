import pints
import ionbench
import numpy as np
from ionbench.optimisers.pints_optimisers import classes_pints


def run(bm, x0=[], maxIter=1000, debug=False):
    """
    Runs Nelder Mead from Pints using a benchmarker.

    Parameters
    ----------
    bm : Benchmarker
        A benchmarker to evaluate the performance of the optimisation algorithm.
    x0 : list, optional
        Initial parameter vector from which to start optimisation. Default is [], in which case a randomly sampled parameter vector is retrieved from bm.sample().
    maxIter : int, optional
        Number of iterations of Nelder Mead to run. The default is 1000.
    debug : bool, optional
        If True, logging messages are printed every iteration. Otherwise the default of every iteration for the first 3 and then every 20 iterations. The default is False.

    Returns
    -------
    xbest : list
        The best parameters identified by Nelder-Mead.

    """
    if len(x0) == 0:
        x0 = bm.sample()
    model = classes_pints.Model(bm)
    problem = pints.SingleOutputProblem(model, np.arange(0, model.bm.tmax, model.bm.freq), model.bm.data)
    error = pints.RootMeanSquaredError(problem)

    # Create an optimisation controller
    opt = pints.OptimisationController(error, x0, method=pints.NelderMead)
    opt.set_max_iterations(maxIter)
    if debug:
        opt.set_log_interval(iters=1)
    # Run the optimisation
    x, f = opt.run()

    model.bm.evaluate(x)
    return x


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


if __name__ == '__main__':
    bm = ionbench.problems.staircase.HH_Benchmarker()
    mod = get_modification()
    mod.apply(bm)
    run(bm, debug=True, **mod.kwargs)

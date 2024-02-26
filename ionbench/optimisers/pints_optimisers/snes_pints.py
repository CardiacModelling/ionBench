import pints
import ionbench
import numpy as np
from ionbench.optimisers.pints_optimisers import classes_pints


def run(bm, x0=None, maxIter=1000, debug=False):
    """
    Runs SNES (Separable Natural Evolution Strategy) from Pints using a benchmarker.

    Parameters
    ----------
    bm : Benchmarker
        A benchmarker to evaluate the performance of the optimisation algorithm.
    x0 : list, optional
        Initial parameter vector from which to start optimisation. Default is None, in which case a randomly sampled parameter vector is retrieved from bm.sample().
    maxIter : int, optional
        Number of iterations of SNES to run. The default is 1000.
    debug : bool, optional
        If True, logging messages are printed every iteration. Otherwise, the default of every iteration for the first 3 and then every 20 iterations. The default is False.

    Returns
    -------
    xbest : list
        The best parameters identified by SNES.

    """
    if x0 is None:
        x0 = bm.sample()
    model = classes_pints.Model(bm)
    problem = pints.SingleOutputProblem(model, np.arange(0, model.bm.tmax, model.bm.freq), model.bm.data)
    error = pints.RootMeanSquaredError(problem)

    if bm._parameters_bounded:
        if bm._rates_bounded:
            boundaries = classes_pints.AdvancedBoundaries(bm)
        else:
            boundaries = pints.RectangularBoundaries(bm.input_parameter_space(bm.lb), bm.input_parameter_space(bm.ub))
        counter = 1
        while not boundaries.check(x0):
            x0 = bm.sample()
            counter += 1
        if counter > 10:
            print(f'Struggled to find parameters in bounds. Required {counter} iterations.')
        opt = pints.OptimisationController(error, x0, method=pints.SNES, boundaries=boundaries)
    else:
        opt = pints.OptimisationController(error, x0, method=pints.SNES)
    opt.set_max_iterations(maxIter)
    if debug:
        opt.set_log_interval(iters=1)
    # Run the optimisation
    x, f = opt.run()

    model.bm.evaluate()
    return x


def get_modification(modNum=1):
    """
    No modification for this optimiser. Will use an empty modification.

    Returns
    -------
    mod : modification
        Empty modification

    """
    mod = ionbench.modification.Empty(name='snes_pints')
    return mod


if __name__ == '__main__':
    bm = ionbench.problems.staircase.HH()
    mod = get_modification()
    mod.apply(bm)
    run(bm, debug=True, **mod.kwargs)

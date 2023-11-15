import pints
import ionbench
import numpy as np
from ionbench.optimisers.pints_optimisers import classes_pints


def run(bm, x0=[], maxIter=1000):
    """
    Runs SNES (Seperable Natural Evolution Strategy) from Pints using a benchmarker.

    Parameters
    ----------
    bm : Benchmarker
        A benchmarker to evaluate the performance of the optimisation algorithm.
    x0 : list, optional
        Initial parameter vector from which to start optimisation. Default is [], in which case a randomly sampled parameter vector is retrieved from bm.sample().
    maxIter : int, optional
        Number of iterations of SNES to run. The default is 1000.

    Returns
    -------
    xbest : list
        The best parameters identified by SNES.

    """
    if len(x0) == 0:
        x0 = bm.sample()
    model = classes_pints.Model(bm)
    problem = pints.SingleOutputProblem(model, np.arange(model.bm.tmax), model.bm.data)
    error = pints.RootMeanSquaredError(problem)

    # Create an optimisation controller
    if bm._bounded:
        boundaries = pints.RectangularBoundaries(bm.input_parameter_space(bm.lb), bm.input_parameter_space(bm.ub))
        opt = pints.OptimisationController(error, x0, method=pints.SNES, boundaries=boundaries)
    else:
        opt = pints.OptimisationController(error, x0, method=pints.SNES)
    opt.set_max_iterations(maxIter)
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
    mod = ionbench.modification.Empty(name='snes_pints')
    return mod


if __name__ == '__main__':
    bm = ionbench.problems.staircase.HH_Benchmarker()
    mod = get_modification()
    mod.apply(bm)
    run(bm)

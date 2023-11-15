import pints
import ionbench
import numpy as np
from ionbench.optimisers.pints_optimisers import classes_pints
# Limited information on the implementation given in Clausen 2020.


def run(bm, x0=[], maxIter=1000):
    """
    Runs PSO (Particle Swarm Optimisation) from Pints and then Nelder Mead from Pints.

    Parameters
    ----------
    bm : Benchmarker
        A benchmarker to evaluate the performance of the optimisation algorithm.
    x0 : list, optional
        Initial parameter vector from which to start optimisation. Default is [], in which case a randomly sampled parameter vector is retrieved from bm.sample().
    maxIter : int, optional
        Number of iterations of PSO and Nelder Mead to run. Maximum total number of iterations is 2*maxIter. The default is 1000.

    Returns
    -------
    xbest : list
        The best parameters identified by PSO.

    """
    if len(x0) == 0:
        x0 = bm.sample()
    model = classes_pints.Model(bm)
    problem = pints.SingleOutputProblem(model, np.arange(model.bm.tmax), model.bm.data)
    error = pints.RootMeanSquaredError(problem)
    bounds = pints.RectangularBoundaries(bm.input_parameter_space(bm.lb), bm.input_parameter_space(bm.ub))
    # Create an optimisation controller
    opt = pints.OptimisationController(error, x0, method=pints.PSO, boundaries=bounds)
    opt.set_max_iterations(maxIter)
    # Run the optimisation
    x, f = opt.run()
    # Create an optimisation controller
    opt = pints.OptimisationController(error, x0, method=pints.NelderMead)
    opt.set_max_iterations(maxIter)
    # Run the optimisation
    x, f = opt.run()

    model.bm.evaluate(x)
    return x


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
    bm = ionbench.problems.staircase.HH_Benchmarker()
    mod = get_modification()
    mod.apply(bm)
    run(bm, maxIter=20, debug=True, **mod.kwargs)

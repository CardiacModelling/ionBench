import pints
from ionBench.problems import staircase
import numpy as np
from ionBench.optimisers.pints_optimisers import classes_pints

def run(bm, logTransforms = [], iterCount=1, maxIter=1000):
    """
    Runs XNES (Exponential Natural Evolution Strategy) from Pints using a benchmarker. 

    Parameters
    ----------
    bm : Benchmarker
        A benchmarker to evaluate the performance of the optimisation algorithm.
    logTransforms : list, optional
        List of parameter indices to log transforms. The default is [], so no parameters should be log-transformed.
    iterCount : int, optional
        Number of times to repeat the algorithm. The default is 1.
    maxIter : int, optional
        Number of iterations of CMA-ES to run per repeat. The default is 1000.

    Returns
    -------
    xbest : list
        The best parameters identified by XNES.

    """
    parameters = np.ones(bm.n_parameters())
    model = classes_pints.Model(bm)
    problem = pints.SingleOutputProblem(model, np.arange(model.bm.tmax), model.bm.data)
    error = pints.RootMeanSquaredError(problem)
    transformation = classes_pints.logTransforms(logTransforms, len(parameters))
    
    fbest = np.inf
    for i in range(iterCount):
        x0 = parameters * 2**np.random.normal(0, 0.5, len(parameters))
        # Create an optimisation controller
        opt = pints.OptimisationController(error, x0, transformation=transformation, method=pints.XNES)
        opt.set_max_iterations(maxIter)
        # Run the optimisation
        x, f = opt.run()
        if f<fbest:
            fbest = f
            xbest = x
    
    model.bm.evaluate(xbest)
    return xbest

if __name__ == '__main__':
    iterCount = 1
    maxIter = 1000
    bm = staircase.HH_Benchmarker()
    logTransforms = [0, 2, 4, 6]
    run(bm = bm, logTransforms = logTransforms, iterCount = iterCount, maxIter = maxIter)
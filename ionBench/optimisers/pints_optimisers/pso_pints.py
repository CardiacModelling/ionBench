import pints
from ionBench.problems import staircase
import numpy as np
from ionBench.optimisers.pints_optimisers import classes_pints

def run(bm, x0 = [], iterCount=1, maxIter=1000):
    """
    Runs PSO (Particle Swarm Optimisation) from Pints using a benchmarker. 

    Parameters
    ----------
    bm : Benchmarker
        A benchmarker to evaluate the performance of the optimisation algorithm.
    x0 : list, optional
        Initial parameter vector from which to start optimisation. Default is [], in which case a randomly sampled parameter vector is retrieved from bm.sample().
    iterCount : int, optional
        Number of times to repeat the algorithm. The default is 1.
    maxIter : int, optional
        Number of iterations of PSO to run per repeat. The default is 1000.

    Returns
    -------
    xbest : list
        The best parameters identified by PSO.

    """
    if x0 == []:
        parameters = bm.sample()
    else:
        parameters = x0
    model = classes_pints.Model(bm)
    problem = pints.SingleOutputProblem(model, np.arange(model.bm.tmax), model.bm.data)
    error = pints.RootMeanSquaredError(problem)
    
    fbest = np.inf
    for i in range(iterCount):
        x0 = parameters * 2**np.random.normal(0, 0.5, len(parameters))
        # Create an optimisation controller
        opt = pints.OptimisationController(error, x0, method=pints.PSO)
        opt.set_max_iterations(maxIter)
        # Run the optimisation
        x, f = opt.run()
        if f<fbest:
            fbest = f
            xbest = x
    
    model.bm.evaluate(xbest)
    return xbest

if __name__ == '__main__':
    bm = staircase.HH_Benchmarker()
    bm.logTransform([True, False]*4+[False])
    run(bm)
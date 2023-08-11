import pints
from ionBench.problems import staircase
import numpy as np
from ionBench.optimisers.pints_optimisers import classes_pints

def run(bm, kCombinations, localBounds = [], logTransforms = [], iterCount=1, maxIter=1000):
    """
    Runs CMA-ES from Pints using a benchmarker. 

    Parameters
    ----------
    bm : Benchmarker
        A benchmarker to evaluate the performance of the optimisation algorithm.
    kCombinations : list
        kCombinations = [[0,1],[4,5]] means param[0]*exp(param[1]*V) and param[4]*exp(param[5]*V) satisfy bounds.
    localBounds : list, optional
        Bounds on the parameters are specified in localBounds. For example, localBounds = [[0,1e-7,1e3],[3,1e-3,1e5]] sets the bounds for parameter index 0 to be [1e-7,1e3] and index 3 to be [1e-3,1e5]. The default is [], so no bounds on parameters.
    logTransforms : list, optional
        List of parameter indices to log transforms. The default is [], so no parameters should be log-transformed.
    iterCount : int, optional
        Number of times to repeat the algorithm. The default is 1.
    maxIter : int, optional
        Number of iterations of CMA-ES to run per repeat. The default is 1000.

    Returns
    -------
    xbest : list
        The best parameters identified by CMA-ES.

    """
    parameters = np.ones(bm.n_parameters())
    model = classes_pints.Model(bm)
    problem = pints.SingleOutputProblem(model, np.arange(model.bm.tmax), model.bm.data)
    error = pints.RootMeanSquaredError(problem)
    transformation = classes_pints.logTransforms(logTransforms, len(parameters))
    boundaries = classes_pints.AdvancedBoundaries(paramCount = model.n_parameters(), localBounds = localBounds, kCombinations = kCombinations, bm = model.bm)
    
    fbest = np.inf
    for i in range(iterCount):
        x0 = parameters * 2**np.random.normal(0, 0.5, len(parameters))
        counter = 1
        while not boundaries.check(x0):
            x0 = parameters * 2**np.random.normal(0, 0.5, len(parameters))
            counter += 1
        if counter > 10:
            print("Struggled to find parameters in bounds")
            print("Required "+str(counter)+" iterations")
        # Create an optimisation controller
        opt = pints.OptimisationController(error, x0, transformation=transformation, method=pints.CMAES, boundaries = boundaries)
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
    localBounds = [[0,1e-7,1e3], [1,1e-7,0.4], [2,1e-7,1e3], [3,1e-7,0.4], [4,1e-7,1e3], [5,1e-7,0.4], [6,1e-7,1e3], [7,1e-7,0.4]]
    kCombinations = [[0,1], [2,3], [4,5], [6,7]]
    run(bm = bm, kCombinations = kCombinations, localBounds = localBounds, logTransforms = logTransforms, iterCount = iterCount, maxIter = maxIter)
import pints
from ionBench.problems import staircase
import numpy as np
from ionBench.optimisers.pints_optimisers import classes_pints

def run(bm, x0 = [], maxIter=1000):
    """
    Runs CMA-ES from Pints using a benchmarker. 

    Parameters
    ----------
    bm : Benchmarker
        A benchmarker to evaluate the performance of the optimisation algorithm.
    x0 : list, optional
        Initial parameter vector from which to start optimisation. Default is [], in which case a randomly sampled parameter vector is retrieved from bm.sample().
    maxIter : int, optional
        Number of iterations of CMA-ES to run. The default is 1000.

    Returns
    -------
    xbest : list
        The best parameters identified by CMA-ES.

    """
    if x0 == []:
        x0 = bm.sample()
    model = classes_pints.Model(bm)
    problem = pints.SingleOutputProblem(model, np.arange(model.bm.tmax), model.bm.data)
    error = pints.RootMeanSquaredError(problem)
    if bm._bounded:
        boundaries = classes_pints.AdvancedBoundaries(bm)
    
    if bm._bounded:
        counter = 1
        while not boundaries.check(x0):
            x0 = bm.sample()
            counter += 1
        if counter > 10:
            print("Struggled to find parameters in bounds")
            print("Required "+str(counter)+" iterations")
    # Create an optimisation controller
    if bm._bounded:
        opt = pints.OptimisationController(error, x0, method=pints.CMAES, boundaries = boundaries)
    else:
        opt = pints.OptimisationController(error, x0, method=pints.CMAES)
    opt.set_max_iterations(maxIter)
    # Run the optimisation
    x, f = opt.run()
    
    model.bm.evaluate(x)
    return x

if __name__ == '__main__':
    bm = staircase.HH_Benchmarker()
    bm.logTransform([True, False]*4+[False])
    bm.addBounds([[1e-7]*8+[-np.inf], [1e3,0.4]*4+[np.inf]])
    run(bm)
import pints
from ionBench.problems import staircase
import numpy as np
from ionBench.pints_optimisers import classes_pints

iterCount = 1
maxIter = 1000
bm = staircase.HH_Benchmarker()
logTransforms = [0, 2, 4, 6]
localBounds = [[0,1e-7,1e3], [1,1e-7,0.4], [2,1e-7,1e3], [3,1e-7,0.4], [4,1e-7,1e3], [5,1e-7,0.4], [6,1e-7,1e3], [7,1e-7,0.4]]
kCombinations = [[0,1], [2,3], [4,5], [6,7]]

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

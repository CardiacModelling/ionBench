import pints
from ionBench.problems import staircase
import numpy as np
from ionBench.pints_optimisers import classes_pints

iterCount = 1
maxIter = 1000
bm = staircase.HH_Benchmarker()
logTransforms = [0, 2, 4, 6]

parameters = np.ones(bm.n_parameters())
model = classes_pints.Model(bm)
problem = pints.SingleOutputProblem(model, np.arange(model.bm.tmax), model.bm.data)
error = pints.RootMeanSquaredError(problem)
transformation = classes_pints.logTransforms(logTransforms, len(parameters))

fbest = np.inf
for i in range(iterCount):
    x0 = parameters * 2**np.random.normal(0, 0.5, len(parameters))
    # Create an optimisation controller
    opt = pints.OptimisationController(error, x0, transformation=transformation, method=pints.PSO)
    opt.set_max_iterations(maxIter)
    # Run the optimisation
    x, f = opt.run()
    if f<fbest:
        fbest = f
        xbest = x

model.bm.evaluate(xbest)

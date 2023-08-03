import pints
from ionBench.problems import staircase
import numpy as np

class Model(pints.ForwardModel):
    def __init__(self):
        self.bm = staircase.HH_Benchmarker()
        
    def n_parameters(self):
        return self.bm.n_parameters()
    
    def simulate(self, parameters, times):
        # Reset the simulation
        return self.bm.simulate(parameters, times)

iterCount = 1
maxIter = 1000
model = Model()
problem = pints.SingleOutputProblem(model, np.arange(model.bm.tmax), model.bm.data)
parameters = np.ones(model.n_parameters())
error = pints.RootMeanSquaredError(problem)
logTransforms = [0, 2, 4, 6]

for i in range(len(parameters)):
    if i == 0:
        if i in logTransforms:
            transformation = pints.LogTransformation(n_parameters=1)
        else:
            transformation = pints.IdentityTransformation(n_parameters=1)
    else:
        if i in logTransforms:
            transformation = pints.ComposedTransformation(transformation, pints.LogTransformation(n_parameters=1))
        else:
            transformation = pints.ComposedTransformation(transformation, pints.IdentityTransformation(n_parameters=1))

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

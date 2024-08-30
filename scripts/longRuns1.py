import ionbench
import importlib
from inspect import signature
import numpy as np
import os
import pickle

np.random.seed(1)  # Different seed compared with initial runs
maxRuns = 1000

bm = ionbench.problems.staircase.HH()
bm.plotter = False
initParams = bm.sample(maxRuns)
app = ionbench.APP_UNIQUE[8]  # Wilhelms2012b only

opt = importlib.import_module(app['module'])
mod = opt.get_modification(app['modNum'])
mod.apply(bm)
x0 = [bm.input_parameter_space(x) for x in initParams]
kwargs = app['kwargs'] | mod.kwargs
if 'maxIter' in signature(opt.run).parameters:
    kwargs = kwargs | {'maxIter': 25000}
if 'nGens' in signature(opt.run).parameters:
    kwargs = kwargs | {'nGens': 10000}
ionbench.utils.multistart(opt.run, bm, x0, 'hh_trr_long_significance', **kwargs)

# Save file of times and successes
successes = []
costTimes = []
gradTimes = []
times = []
for runNum in range(maxRuns):
    bm.reset()
    bm.tracker.load(f"hh_trr_long_significance_run{runNum}.pickle")
    i = bm.tracker.when_converged(bm.COST_THRESHOLD)
    i = -1 if i is None else i
    successes.append(bm.tracker.bestCosts[i] < bm.COST_THRESHOLD)
    ct, gt = bm.tracker.costSolves[i], bm.tracker.gradSolves[i]
    totalTime = ct + gt * 8.771391927040481  # Total time, in FEs
    costTimes.append(ct)
    gradTimes.append(gt)
    times.append(totalTime)

data = {'successes': successes, 'costs': costTimes, 'grads': gradTimes, 'times': times}
print(data)
with open('hh_trr_long_significance.pickle', 'wb') as f:
    pickle.dump(data, f)

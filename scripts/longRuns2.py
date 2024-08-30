import ionbench
import importlib
from inspect import signature
import numpy as np
import os
import pickle
from ionbench.utils.results import bootstrap_ERT

np.random.seed(1)  # Different seed compared with initial runs
maxRuns = 1000
batchSize = 20
B = 10000

batch = int(os.environ['SLURM_ARRAY_TASK_ID'])

bm = ionbench.problems.staircase.HH()
bm.plotter = False
initParams = bm.sample(1000)
app = ionbench.APP_UNIQUE[batch]
print(app['module'])
opt = importlib.import_module(app['module'])
mod = opt.get_modification(app['modNum'])
mod.apply(bm)
kwargs = app['kwargs'] | mod.kwargs
if 'maxIter' in signature(opt.run).parameters:
    kwargs = kwargs | {'maxIter': 25000}
if 'nGens' in signature(opt.run).parameters:
    kwargs = kwargs | {'nGens': 10000}
x0 = [bm.input_parameter_space(x) for x in initParams]

# Get TRR data
with open('hh_trr_long_significance.pickle', 'rb') as f:
    data = pickle.load(f)
    successesTRR = data['successes']
    costTimesTRR = data['costs']
    gradTimesTRR = data['grads']
    timesTRR = data['times']

successes = []
costTimes = []
gradTimes = []
times = []

for i in range(maxRuns//batchSize):
    # Run the approach batchSize times
    bm.reset(False)  # Don't reset modification
    ionbench.utils.multistart(opt.run, bm, x0[i*batchSize:(i+1)*batchSize], f"hh_long_run_batch_{i}_{app['module']}modNum{app['modNum']}", **kwargs)

    # Append the data to the current track
    for runNum in range(batchSize):
        bm.reset(False)
        bm.tracker.load(f"hh_long_run_batch_{i}_{app['module']}modNum{app['modNum']}_run{runNum}.pickle")
        j = bm.tracker.when_converged(bm.COST_THRESHOLD)
        j = -1 if j is None else j
        successes.append(bm.tracker.bestCosts[j] < bm.COST_THRESHOLD)
        ct, gt = bm.tracker.costSolves[j], bm.tracker.gradSolves[j]
        totalTime = ct + gt * 8.771391927040481  # Total time, in FEs
        costTimes.append(ct)
        gradTimes.append(gt)
        times.append(totalTime)

    # Check for significance
    sTRR = successesTRR[:len(successes)]
    tTRR = timesTRR[:len(times)]
    ertTRR = np.zeros(B)
    ertApp = np.zeros(B)
    for b in range(B):
        ertTRR[b] = bootstrap_ERT(np.array(sTRR), np.array(tTRR))
        ertApp[b] = bootstrap_ERT(np.array(successes), np.array(times))
    X, Y = np.meshgrid(ertTRR, ertApp)
    p = np.mean(Y < X)
    print('Median ERT TRR:', np.median(ertTRR))
    print('Median ERT App:', np.median(ertApp))
    if p < 0.05:
        print(f"TRR significantly better after {len(successes)}.")
        print(f"Approximate p value: {p}")
        break
    elif p > 0.95:
        print(f"{app['module']} {app['modNum']} significantly better after {len(successes)}.")
        print(f"Approximate p value: {p}")
        break
    else:
        print(f"No significance after {len(successes)}.")
        print(f"Approximate p value: {p}")


data = {'successes': successes, 'costs': costTimes, 'grads': gradTimes, 'times': times}
print(data)
with open(f'hh_long_run_{app["module"]}modNum{app["modNum"]}.pickle', 'wb') as f:
    pickle.dump(data, f)

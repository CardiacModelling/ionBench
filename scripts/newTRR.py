import numpy as np
import os
import ionbench
from ionbench.utils.results import bootstrap_ERT

problem = int(os.environ['SLURM_ARRAY_TASK_ID'])
nPoints = [500, 400, 100, 250, 3000][problem]

# Define modifications
wilhelmsMod = ionbench.modification.Wilhelms2012b()
dokosMod = ionbench.modification.Dokos2004()
newMod = ionbench.modification.Modification(logTransform='on', parameterBounds='on', rateBounds='on')

bms = [ionbench.problems.staircase.HH(sensitivities=True), ionbench.problems.staircase.MM(sensitivities=True), ionbench.problems.loewe2016.IKr(sensitivities=True), ionbench.problems.loewe2016.IKur(sensitivities=True), ionbench.problems.moreno2016.INa(sensitivities=True)]

bm = bms[problem]
appName = 'dokos2004' if 'staircase.hh' == bm.NAME else 'wilhelms2012b'

bm.plotter = False
np.random.seed(2)
initParams = bm.sample(nPoints)
if 'staircase.hh' == bm.NAME:
    dokosMod.apply(bm)
else:
    wilhelmsMod.apply(bm)
x0 = [bm.input_parameter_space(x) for x in initParams]
if 'staircase.hh' == bm.NAME:
    ionbench.utils.multistart(ionbench.optimisers.external_optimisers.curvilinearGD_Dokos2004.run, bm, x0, filename=f'{appName}_{bm.NAME}', maxIter=25000)
else:
    ionbench.utils.multistart(ionbench.optimisers.scipy_optimisers.trustRegionReflective_scipy.run, bm, x0, filename=f'{appName}_{bm.NAME}', maxIter=25000)
bm.reset()
newMod.apply(bm)
x0 = [bm.input_parameter_space(x) for x in initParams]
ionbench.utils.multistart(ionbench.optimisers.scipy_optimisers.trustRegionReflective_scipy.run, bm, x0, filename=f'newTRR_{bm.NAME}')

timeRatios = [8.766738292825124, 13.312410027158347, 10.514951888964069, 26, 17]


def get_data(filename):
    times = []
    successes = []
    for i in range(nPoints):
        bm.tracker.load(f"{filename}_run{i}.pickle")
        i = bm.tracker.when_converged(bm.COST_THRESHOLD)
        i = -1 if i is None else i
        try:
            successes.append(bm.tracker.bestCosts[i] < bm.COST_THRESHOLD)
        except IndexError:
            successes.append(False)
        try:
            t1 = bm.tracker.costSolves[i]
        except IndexError:
            t1 = 0
        try:
            t2 = bm.tracker.gradSolves[i]
        except IndexError:
            t2 = 0
        times.append(t1 + t2 * timeRatios[bms.index(bm)])
        bm.reset()
    return np.array(times), np.array(successes)

bootstrapCount = 10000
print(f"Problem: {bm.NAME}")
tWilhelms, sWilhelms = get_data(f'{appName}_{bm.NAME}')
tNew, sNew = get_data(f'newTRR_{bm.NAME}')

print("Wilhelms")
print("Times")
print(tWilhelms)
print("Successes")
print(sWilhelms)

print("New")
print("Times")
print(tNew)
print("Successes")
print(sNew)

samplesWilhelms = np.zeros(bootstrapCount)
samplesNew = np.zeros(bootstrapCount)
for b in range(bootstrapCount):
    samplesWilhelms[b] = bootstrap_ERT(sWilhelms, tWilhelms)
    samplesNew[b] = bootstrap_ERT(sNew, tNew)
print(f'Median ERT Wilhelms: {np.median(samplesWilhelms)}, New: {np.median(samplesNew)}, STD Wilhelms: {np.std(samplesWilhelms)}, New: {np.std(samplesNew)}')
A, B = np.meshgrid(samplesNew, samplesWilhelms)
print(f'New significantly better with significance {np.mean(A < B)}')

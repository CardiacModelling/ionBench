import ionbench
import matplotlib.pyplot as plt

basePath = 'ionbench.optimisers.'
scipyPath = 'scipy_optimisers.'
externalPath = 'external_optimisers.'
pintsPath = 'pints_optimisers.'
tmp = ['lm', 'nelderMead', 'powell', 'trustRegionReflective']
scipyOpt = [scipyPath+s+'_scipy' for s in tmp]
tmp = ['cmaes', 'nelderMead', 'pso', 'snes', 'xnes']
pintsOpt = [pintsPath+s+'_pints' for s in tmp]
tmp = ['GA_Bot2012', 'GA_Smirnov2020', 'patternSearch_Kohjitani2022', 'ppso_Chen2012']
extOpt = [externalPath+s for s in tmp]

opts = [basePath + s for s in (scipyOpt + pintsOpt + extOpt)]
print(opts)
bm = ionbench.problems.staircase.HH_Benchmarker()

for st in opts:
    bm.tracker.load('tracker_'+st+'.pickle')
    plt.figure()
    plt.title(st)
    bm.tracker.plot()

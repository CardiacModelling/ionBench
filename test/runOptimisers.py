import ionbench
import importlib

basePath = 'ionbench.optimisers.'
scipyPath = 'scipy_optimisers.'
externalPath = 'external_optimisers.'
pintsPath = 'pints_optimisers.'
tmp = ['lm', 'nelderMead', 'powell', 'trustRegionReflective']
scipyOpt = [scipyPath+s+'_scipy' for s in tmp]
tmp = ['cmaes', 'nelderMead', 'pso', 'snes', 'xnes']
pintsOpt = [pintsPath+s+'_pints' for s in tmp]
tmp = ['GA_Bot2012', 'GA_Smirnov2020', 'hybridPSOTRR_Loewe2016', 'patternSearch_Kohjitani2022', 'ppso_Chen2012']
extOpt = [externalPath+s for s in tmp]

opts = [basePath + s for s in (scipyOpt + pintsOpt + extOpt)]
print(opts)
bm = ionbench.problems.staircase.HH_Benchmarker()

for st in opts:
    print(st)
    opt = importlib.import_module(st)
    app = opt.get_approach()
    app.apply(bm)
    out = opt.run(bm)
    print(out)
    bm.reset()
    
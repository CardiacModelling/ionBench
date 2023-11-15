import ionbench
import importlib

bm = ionbench.problems.staircase.HH_Benchmarker()
initParams = bm.sample(n=5)

for app in ionbench.APP_ALL:
    print(app['module'])
    opt = importlib.import_module(app['module'])
    mod = opt.get_modification(app['modNum'])
    mod.apply(bm)
    ionbench.multistart(opt.run, bm, initParams, app['module'] + 'modNum' + str(app['modNum']), **app['kwargs'])
    bm.reset()

import ionbench
import importlib
import numpy as np

bm = ionbench.problems.staircase.HH()
np.random.seed(0)
initParams = bm.sample(n=5)
print(initParams)
for app in ionbench.APP_UNIQUE:
    print(app['module'])
    try:
        opt = importlib.import_module(app['module'])
        mod = opt.get_modification(app['modNum'])
        mod.apply(bm)
        x0 = [bm.input_parameter_space(x) for x in initParams]
        kwargs = app['kwargs'] | mod.kwargs
        ionbench.utils.multistart(opt.run, bm, x0, app['module'] + 'modNum' + str(app['modNum']), **kwargs)
    except Exception as e:
        print(e)
        print(f"Approach {app['module']} with modNum {app['modNum']} failed. Will skip and keep going to get as many results as possible")
    bm.reset()

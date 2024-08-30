import ionbench
import importlib
from inspect import signature
import numpy as np
import os

np.random.seed(0)

batch = int(os.environ['SLURM_ARRAY_TASK_ID'])

bm = ionbench.problems.staircase.HH()
bm.plotter = False
initParams = bm.sample(10)
print(initParams)
app = ionbench.APP_UNIQUE[batch]
print(app['module'])
opt = importlib.import_module(app['module'])
mod = opt.get_modification(app['modNum'])
mod.apply(bm)
x0 = [bm.input_parameter_space(x) for x in initParams]
kwargs = app['kwargs'] | mod.kwargs
if 'maxIter' in signature(opt.run).parameters:
    kwargs = kwargs | {'maxIter': 25000}
if 'nGens' in signature(opt.run).parameters:
    kwargs = kwargs | {'nGens': 10000}
ionbench.utils.multistart(opt.run, bm, x0, 'hh_' + app['module'] + 'modNum' + str(app['modNum']), **kwargs)

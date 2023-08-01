import numpy as np
from ionBench import benchmarker
import scipy.optimize

try: bm
except NameError:
    print('No benchmarker loaded. Creating a new one')
    bm = benchmarker.HH_Benchmarker()
else: bm.reset()

x0 = np.ones(bm.n_parameters())

out = scipy.optimize.minimize(bm.cost, x0, method='powell', options={'disp': True, 'xtol': 1e-4, 'ftol': 1e-4, 'maxiter': 5000, 'maxfev': 20000}, bounds = [(0,None)]*bm.n_parameters())

bm.evaluate(out.x)


import numpy as np
from ionBench import benchmarker
import scipy.optimize

try: bm
except NameError:
    print('No benchmarker loaded. Creating a new one')
    bm = benchmarker.HH_Benchmarker()
else: bm.reset()

x0 = np.ones(bm.n_parameters())

out = scipy.optimize.least_squares(bm.signedError, x0, method='trf', diff_step=1e-3, verbose=2, bounds = ([0]*bm.n_parameters(),[np.inf]*bm.n_parameters()))
#Add bounds
bm.evaluate(out.x)


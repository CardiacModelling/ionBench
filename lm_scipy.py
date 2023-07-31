import numpy as np
import benchmarker
import scipy.optimize

try: bm
except NameError:
    print('No benchmarker loaded. Creating a new one')
    bm = benchmarker.HH_Benchmarker()
else: bm.reset()

x0 = np.ones(bm.n_parameters())

out = scipy.optimize.least_squares(bm.signedError, x0, method='lm', diff_step=1e-3, verbose=1)

bm.evaluate(out.x)


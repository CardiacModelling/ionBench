import ionBench.problems.staircase
import scipy.optimize
import numpy as np

def run(bm, x0, diff_step = 1e-3, bounds = [], maxfev = 20000):
    if bounds == []:
        out = scipy.optimize.least_squares(bm.signedError, x0, method='trf', diff_step=diff_step, verbose=2, max_nfev = maxfev)
    else:
        out = scipy.optimize.least_squares(bm.signedError, x0, method='trf', diff_step=diff_step, verbose=2, max_nfev = maxfev, bounds = bounds)
    
    bm.evaluate(out.x)
    return out.x

if __name__ == '__main__':
    bm = ionBench.problems.staircase.HH_Benchmarker()
    bounds = ([0]*bm.n_parameters(),[np.inf]*bm.n_parameters())
    run(bm = bm, x0 = bm.defaultParams, bounds = bounds)
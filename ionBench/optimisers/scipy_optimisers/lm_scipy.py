import ionBench.problems.staircase
import scipy.optimize

def run(bm, x0, diff_step = 1e-3, maxfev = 20000):
    out = scipy.optimize.least_squares(bm.signedError, x0, method='lm', diff_step=diff_step, verbose=1, max_nfev = maxfev)
    
    bm.evaluate(out.x)
    return out.x

if __name__ == '__main__':
    bm = ionBench.problems.staircase.HH_Benchmarker()
    run(bm = bm, x0 = bm.defaultParams)
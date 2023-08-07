import ionBench.problems.staircase
import scipy.optimize

def run(bm, x0, xatol = 1e-4, fatol = 1e-4, maxiter = 5000, maxfev = 20000, bounds = []):
    if bounds == []:
        out = scipy.optimize.minimize(bm.cost, x0, method='nelder-mead', options={'disp': True, 'xatol': xatol, 'fatol': fatol, 'maxiter': maxiter, 'maxfev': maxfev})
    else:
        out = scipy.optimize.minimize(bm.cost, x0, method='nelder-mead', options={'disp': True, 'xatol': xatol, 'fatol': fatol, 'maxiter': maxiter, 'maxfev': maxfev}, bounds = bounds)
    
    bm.evaluate(out.x)
    return out.x

if __name__ == '__main__':
    bm = ionBench.problems.staircase.HH_Benchmarker()
    bounds = [(0,None)]*bm.n_parameters()
    run(bm = bm, x0 = bm.defaultParams, bounds = bounds)
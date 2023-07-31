import numpy as np
import benchmarker
import scipy.optimize
"""
Solver error for SLSQP
ArithmeticError: Function CVode() failed with flag -4 CV_CONV_FAILURE: Convergence test failures occurred too many times during one internal time step or minimum step size was reached.
SimulationError: A numerical error occurred during simulation at t = 0.
Last reached state: 
  ikr.act = 0.0
  ikr.rec = 1.0
Inputs for binding:
  time        = 0.0
  realtime    = 0.0
  evaluations = 53.0
  pace = -80.0
Function CVode() failed with flag -4 CV_CONV_FAILURE: Convergence test failures occurred too many times during one internal time step or minimum step size was reached.
"""
#TODO:
#Fix SLSQP error above
#I think it isn't recording the derivative cost function evaluations

try: bm
except NameError:
    print('No benchmarker loaded. Creating a new one')
    bm = benchmarker.HH_Benchmarker()
else: bm.reset()

x0 = np.ones(bm.n_parameters())

methods = ['nelder-mead', 'powell', 'CG', 'BFGS', 'Newton-CG', 'L-BFGS-B', 'TNC', 'COBYLA', 'trust-constr'] #trust region methods (except strust-constr) and dogleg require hessian so are ignored, newton cg requires jocobian which is approximated, SLSQP causes solver error - need to investigate

for i in range(len(methods)):
    bm.reset()
    print(methods[i])
    if methods[i] in ['Newton-CG']:
        out = scipy.optimize.minimize(bm.cost, x0, method=methods[i], options={'disp': True}, jac = lambda x : scipy.optimize.approx_fprime(x, bm.cost, epsilon = 1e-3), bounds = [(0,None)]*bm.n_parameters())
    else:
        out = scipy.optimize.minimize(bm.cost, x0, method=methods[i], options={'disp': True}, bounds = [(0,None)]*bm.n_parameters())
    bm.evaluate(out.x)


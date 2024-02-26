import ionbench.problems.staircase
import scipy.optimize
import numpy as np


def run(bm, x0=[], ftol=1e-6, maxIter=1000, debug=False):
    """
    Runs Sequential Least SQuares Programming optimiser from Scipy. An example of a Sequential Quadratic Programming method which uses a quasi-newton update strategy to approximate the hessian.

    Parameters
    ----------
    bm : Benchmarker
        A benchmarker to evaluate the performance of the optimisation algorithm.
    x0 : list, optional
        Initial parameter vector from which to start optimisation. Default is [], in which case a randomly sampled parameter vector is retrieved from bm.sample().
    ftol : float, optional
        Tolerance in for the cost function. Cost function must be less than ftol before algorithm successfully terminates. The default is 0.05.
    maxIter : int, optional
        Maximum number of iterations of SLSQP to use. The default is 1000.

    Returns
    -------
    xbest : list
        The best parameters identified by SLSQP.

    """
    if len(x0) == 0:
        x0 = bm.sample()
        if debug:
            print('Sampling x0')
            print(x0)

    def grad(p):
        return bm.grad(p)

    def cost(p):
        return bm.cost(p)

    if bm._parameters_bounded:
        lb = bm.input_parameter_space(bm.lb)
        ub = bm.input_parameter_space(bm.ub)
        bounds = []
        for i in range(bm.n_parameters()):
            if lb[i] == np.inf:
                lb[i] = None
            if ub[i] == np.inf:
                ub[i] = None
            bounds.append((lb[i], ub[i]))
        if debug:
            print('Bounds transformed')
            print('Old Bounds:')
            print(bm.lb)
            print(bm.ub)
            print('New bounds')
            print(bounds)
        out = scipy.optimize.minimize(cost, x0, jac=grad, method='SLSQP', options={'disp': debug, 'ftol': ftol, 'maxiter': maxIter}, bounds=bounds)
    else:
        out = scipy.optimize.minimize(cost, x0, jac=grad, method='SLSQP', options={'disp': debug, 'ftol': ftol, 'maxiter': maxIter})

    if debug:
        print(f'Cost of {out.fun} found at:')
        print(out.x)

    bm.evaluate()
    return out.x


def get_modification(modNum=1):
    """
    modNum = 1 -> BuenoOrovio2008

    Returns
    -------
    mod : modification
        Modification corresponding to inputted modNum. Default is modNum = 1, so BuenoOrovio2008.

    """
    mod = ionbench.modification.BuenoOrovio2008()
    return mod


if __name__ == '__main__':
    bm = ionbench.problems.staircase.HH(sensitivities=True)
    mod = get_modification()
    mod.apply(bm)
    run(bm, maxIter=50, debug=True, **mod.kwargs)

import ionbench.problems.staircase
import scipy.optimize
import numpy as np


def run(bm, x0=[], ftol=1e-6, maxIter=1000):
    """
    Runs Sequential Least SQuares Programming optimiser from Scipy. An example of a Sequantial Quadratic Programming method which uses a quasi-newton update strategy to approximate the hessian.

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

    def grad(p):
        return bm.grad(p)

    def cost(p):
        return bm.cost(p)

    if bm._bounded:
        lb = bm.lb[:]  # Generate copy
        ub = bm.ub[:]  # Generate copy
        bounds = []
        for i in range(bm.n_parameters()):
            if lb[i] == np.inf:
                lb[i] = None
            if ub[i] == np.inf:
                ub[i] = None
            bounds.append((lb[i], ub[i]))

    out = scipy.optimize.minimize(cost, x0, jac=grad, method='SLSQP', options={'disp': True, 'ftol': ftol, 'maxiter': maxIter}, bounds=bounds)

    bm.evaluate(out.x)
    return out.x


def get_modification():
    """

    Returns
    -------
    mod : modification
        Modification of Bueno-Orovio et al 2008.
    """
    mod = ionbench.modification.BuenoOrovio2008()
    return mod


if __name__ == '__main__':
    bm = ionbench.problems.staircase.HH_Benchmarker(sensitivities=True)
    mod = get_modification()
    mod.apply(bm)
    run(bm, maxIter=50)

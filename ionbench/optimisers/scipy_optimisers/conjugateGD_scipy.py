import ionbench.problems.staircase
import scipy.optimize
import numpy as np


def run(bm, x0=[], gtol=0.001, maxiter=1000):
    """
    Runs Conjugate Gradient Descent optimiser from Scipy. Bounds are automatically loaded from the benchmarker if present.

    Parameters
    ----------
    bm : Benchmarker
        A benchmarker to evaluate the performance of the optimisation algorithm.
    x0 : list, optional
        Initial parameter vector from which to start optimisation. Default is [], in which case a randomly sampled parameter vector is retrieved from bm.sample().
    gtol : float, optional
        Tolerance in for the gradient. Gradient norm must be less than gtol before algorithm successfully terminates. The default is 0.001.
    maxiter : int, optional
        Maximum number of iterations of Conjugate Gradient Descent to use. The default is 1000.

    Returns
    -------
    xbest : list
        The best parameters identified by Conjugate Gradient Descent.

    """
    if len(x0) == 0:
        x0 = bm.sample()

    def fun(p):
        return bm.grad(p, returnCost=True)

    if bm._bounded:
        lb = bm.lb[:]  # Generate copy
        ub = bm.ub[:]  # Generate copy
        for i in range(bm.n_parameters()):
            if lb[i] == np.inf:
                lb[i] = None
            if ub[i] == np.inf:
                ub[i] = None
        bounds = (lb, ub)

        out = scipy.optimize.minimize(fun, x0, jac=True, method='CG', options={
                                      'disp': True, 'gtol': gtol, 'maxiter': maxiter}, bounds=bounds)
    else:
        out = scipy.optimize.minimize(fun, x0, jac=True, method='CG', options={
                                      'disp': True, 'gtol': gtol, 'maxiter': maxiter})

    bm.evaluate(out.x)
    return out.x


if __name__ == '__main__':
    bm = ionbench.problems.staircase.HH_Benchmarker(sensitivities=True)
    bm._useScaleFactors = True
    x0 = bm.sample()
    run(bm, x0)


def get_modification(modNum=1):
    """
    modNum = 1 -> Sachse2003
    modNum = 2 -> Vanier1999

    Returns
    -------
    mod : modification
        Modification corresponding to inputted modNum. Default is modNum = 1, so Sachse2003.

    """

    if modNum == 1:
        mod = ionbench.modification.Sachse2003()
    elif modNum == 2:
        mod = ionbench.modification.Vanier1999()
    else:
        mod = ionbench.modification.Empty(name='conjugateGD_scipy_scipy')
    return mod

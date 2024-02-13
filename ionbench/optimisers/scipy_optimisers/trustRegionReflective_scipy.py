import ionbench.problems.staircase
import scipy.optimize


def run(bm, x0=[], diff_step=1e-3, maxIter=1000, debug=False):
    """
    Runs Trust Region Reflective optimiser from Scipy.

    Parameters
    ----------
    bm : Benchmarker
        A benchmarker to evaluate the performance of the optimisation algorithm.
    x0 : list, optional
        Initial parameter vector from which to start optimisation. Default is [], in which case a randomly sampled parameter vector is retrieved from bm.sample().
    diff_step : float, optional
        Step size for finite difference calculation. The default is 1e-3.
    maxIter : int, optional
        Maximum number of cost function evaluations. The default is 1000.

    Returns
    -------
    xbest : list
        The best parameters identified by Trust Region Reflective.

    """
    def grad(p):
        return bm.grad(p, residuals=True)

    if len(x0) == 0:
        x0 = bm.sample()
        if debug:
            print('Sampling x0')
            print(x0)

    if debug:
        verbose = 2
    else:
        verbose = 1

    if bm._parameters_bounded:
        bounds = (bm.lb, bm.ub)
        out = scipy.optimize.least_squares(bm.signed_error, x0, method='trf', jac=grad, verbose=verbose, max_nfev=maxIter, bounds=bounds)
    else:
        out = scipy.optimize.least_squares(bm.signed_error, x0, method='trf', jac=grad, verbose=verbose, max_nfev=maxIter)

    if debug:
        print(f'Cost of {out.cost} found at:')
        print(out.x)

    bm.evaluate(out.x)
    return out.x


def get_modification(modNum=1):
    """
    modNum = 1 -> Du2014
    modNum = 2 -> Loewe2016
    modNum = 3 -> Wilhelms2012b

    Returns
    -------
    mod : modification
        Modification corresponding to inputted modNum. Default is modNum = 1, so Du2014.

    """

    if modNum == 1:
        mod = ionbench.modification.Du2014()
    elif modNum == 2:
        mod = ionbench.modification.Loewe2016()
    elif modNum == 3:
        mod = ionbench.modification.Wilhelms2012b()
    else:
        mod = ionbench.modification.Empty(name='trr_scipy')
    return mod


if __name__ == '__main__':
    bm = ionbench.problems.staircase.HH(sensitivities=True)
    mod = get_modification()
    mod.apply(bm)
    run(bm, debug=True, **mod.kwargs)

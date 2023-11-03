import ionbench.problems.staircase
import scipy.optimize


def run(bm, x0=[], diff_step=1e-3, maxIter=1000):
    """
    Runs lm (Levenberg-Marquardt) least squares optimiser from Scipy.

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
        The best parameters identified by LM.

    """
    if len(x0) == 0:
        x0 = bm.sample()

    out = scipy.optimize.least_squares(bm.signed_error, x0, method='lm', diff_step=diff_step, verbose=1, max_nfev=maxIter)

    bm.evaluate(out.x)
    return out.x


if __name__ == '__main__':
    bm = ionbench.problems.staircase.HH_Benchmarker()
    run(bm)


def get_modification(modNum=1):
    """
    modNum = 1 -> Balser1990
    modNum = 2 -> Clancy1999

    Returns
    -------
    mod : modification
        Modification corresponding to inputted modNum. Default is modNum = 1, so Balser1990.

    """

    if modNum == 1:
        mod = ionbench.modification.Balser1990()
    elif modNum == 2:
        mod = ionbench.modification.Clancy1999()
    else:
        mod = ionbench.modification.Empty(name='lm_scipy')
    return mod

"""
This module describes the pattern search algorithm from Kohjitani et al. 2022.
This algorithm is a simple pattern search algorithm that explores neighbouring points around a base point, and moves to the best neighbouring point if an improvement is found.
The step size is reduced if no improvements are found in the neighbouring points.
The algorithm terminates when the step size is sufficiently small.
"""
import ionbench
import numpy as np


# noinspection PyShadowingNames
def run(bm, x0=None, maxIter=1000, CrtStp=2e-5, Stp=1 / 100, RedFct=1 / 4, maxfev=20000, debug=False):
    """
    Runs the pattern search algorithm from Kohjitani et al. 2022.

    Parameters
    ----------
    bm : Benchmarker
        A benchmarker to evaluate the performance of the optimisation algorithm.
    x0 : list, optional
        Initial parameter vector from which to start optimisation. Default is None, in which case a randomly sampled parameter vector is retrieved from bm.sample().
    maxIter : int, optional
        Maximum number of iterations before termination.
    CrtStp : float, optional
        The minimum step size. If there are no improvements within CrtStp, the optimisation terminates. The default is 2e-5.
    Stp : float, optional
        Initial step size. The default is 1/100.
    RedFct : float, optional
        The reduction factor. If the center point is better than its neighbours, the step size is scaled by this reduction factor. The default is 1/4.
    maxfev : int, optional
        Maximum number of cost function evaluations. The default is 20000.
    debug : bool, optional
        If True, debug information will be printed, reporting that status of the optimisation each generation. The default is False.

    Returns
    -------
    xbest : list
        The best parameters identified.

    """
    cost_func = ionbench.utils.cache.get_cached_cost(bm)

    funcCounter = 0
    iterCounter = 0

    # noinspection PyShadowingNames
    def explore(BP, Stp):
        """
        Explores neighbouring points to the base point BP.
        Parameters
        ----------
        BP : numpy array
            The base point around which to explore. Check cost by increasing and decreasing each parameter by Stp.
        Stp : float
            The step size.

        Returns
        -------
        foundImprovement : bool
            True if an improvement is found, False otherwise.
        NP : numpy array
            The new base point, if an improvement is found. Otherwise, the same as BP.
        """
        foundImprovement = False
        NP = np.copy(BP)
        MSE = cost_func(tuple(BP))  # No real computation cost from this thanks to the cache
        for i in range(bm.n_parameters()):
            home = BP[i]
            NP[i] = home + Stp  # Positive Step
            MSEp = cost_func(tuple(NP))  # Positive MSE
            NP[i] = home - Stp  # Negative Step
            MSEn = cost_func(tuple(NP))  # Negative MSE
            minMSE = min(MSEp, MSEn)  # MSE in the best direction (positive or negative Stp)
            if minMSE < MSE:  # If improvement found
                if MSEp < MSEn:  # If positive step is better
                    NP[i] = home + Stp  # Take positive step
                else:  # If negative step is better
                    NP[i] = home - Stp  # Take negative step
                MSE = minMSE  # Either way, record new MSE
                foundImprovement = True
            else:  # If no improvement
                NP[i] = home  # Restore center point
        return foundImprovement, NP

    if x0 is None:
        x0 = bm.sample()

    BP = x0  # Set initial base point
    NP = np.copy(BP)
    funcCounter += 1
    while Stp > CrtStp:  # Stop when step size is sufficiently small
        # Explore neighbouring points
        if debug:
            print("------------")
            print(f'Current step size: {Stp}')
            print(f'Cost: {cost_func(tuple(BP))}')
        improvementFound, NP = explore(BP, Stp)  # Explore neighbouring points
        funcCounter += 4
        while improvementFound:
            iterCounter += 1
            if debug:
                print(f'Improvement Found? {improvementFound}')
            BP = np.copy(NP)  # Move to new improved point
            improvementFound, NP = explore(BP, Stp)  # Explore neighbouring points
            funcCounter += 4
            if funcCounter > maxfev:
                print("Exceeded maximum number of function evaluations.")
                bm.set_max_iter_flag()
                bm.evaluate()
                return NP
            if iterCounter > maxIter:
                print("Exceeded maximum number of iterations.")
                bm.set_max_iter_flag()
                bm.evaluate()
                return NP

            if bm.is_converged():
                bm.evaluate()
                return NP

        Stp *= RedFct  # Decrease step size if all neighbouring points are worse

    bm.evaluate()
    return NP


# noinspection PyUnusedLocal,PyShadowingNames
def get_modification(modNum=1):
    """
    modNum = 1 -> Kohjitani2022

    Returns
    -------
    mod : modification
        Modification corresponding to inputted modNum. Default is modNum = 1, so Kohjitani2022.

    """
    mod = ionbench.modification.Kohjitani2022()
    return mod

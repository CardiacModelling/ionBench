# Find the curved line between the steepest direction and the minimum is a quadratic shape is assumed based on the local curvature. Minimise along this line using Brent's method, as implemented in Press et al. "Numerical Recipes in C" by scipy. Update from Dokos 2004 compared with Dokos 2003 says that if you reach a local minimum whose cost is non-zero, you should weight the residuals and carry on

import numpy as np
import ionbench
import scipy


# noinspection PyShadowingNames
def run(bm, x0=None, maxIter=1000, maxInnerIter=100, costThreshold=0, debug=False):
    """
    Curvilinear gradient descent from Dokos 2004. This method was first introduced in Dokos 2003, with this implementation using the updated scheme which includes weighted residuals (Dokos 2004). It calculates a curved path which initially follows the steepest descent, but curves to finally converge towards a Gauss-Newton step (minimum of a locally defined quadratic). In order to optimise along this line, we use Scipy's implementation of Brent's method (an implementation based on Press et al. "Numerical Recipes in C").

    Parameters
    ----------
    bm : Benchmarker
        A benchmarker to evaluate the performance of the optimisation algorithm.
    x0 : list, optional
        Initial parameter guess. If x0=None (the default), then the population will be sampled from the benchmarker problems .sample() method.
    maxIter : int, optional
        Maximum number of iterations of curvilinear gradient descent, summed up over all reweighting restarts. The default is 1000.
    maxInnerIter : int, optional
        Maximum number of iterations of curvilinear gradient descent before reweighting. The default is 100.
    costThreshold : float, optional
        The maximum acceptable weighted cost. Optimisation terminates only once this threshold is reached, or maxIter is exceeded. Default is 0.
    debug : bool, optional
        If True, debug information will be printed, reporting that status of the optimisation each generation. The default is False.

    Returns
    -------
    xbest : list
        The best parameters identified.

    """

    # Update to the weight vector for the residuals
    # noinspection PyShadowingNames
    def weight_update(wOld, r):
        """
        Update the weights for the residuals.
        Parameters
        ----------
        wOld : numpy array
            The previous weights.
        r : numpy array
            The residuals.

        Returns
        -------
        w : numpy array
            The updated weights.
        """
        rAbs = np.abs(r)
        mu1 = np.random.rand() * min(np.max(rAbs), 2 * np.median(rAbs))
        mu2 = np.random.rand() * min(np.max(rAbs), 2 * np.mean(rAbs))
        mu3 = (0.2 * np.random.rand() + 0.8) * np.max(rAbs)
        sigma = np.std(rAbs)
        dw = np.zeros(len(wOld))
        for i in range(len(dw)):
            dw[i] = np.exp(((rAbs[i] - mu1) / sigma) ** 2) + np.exp(((rAbs[i] - mu2) / sigma) ** 2) + np.exp(
                ((rAbs[i] - mu3) / sigma) ** 2)
        s = min(np.sqrt(10 * np.dot(wOld ** 2, r ** 2)), 1e5) / (np.sqrt(np.dot(dw ** 2, r ** 2)) + 1e-12)
        return wOld + s * dw

    # Set up transform functions to map between upper and lower bounds
    if not bm.parametersBounded:  # pragma: no cover
        bm.add_parameter_bounds()
    ub = bm.input_parameter_space(bm.ub)
    lb = bm.input_parameter_space(bm.lb)

    # noinspection PyShadowingNames
    def fitting_params(p):
        """
        Transform from model solve space to fitting space.
        Parameters
        ----------
        p : numpy array
            The parameters in model solve space.

        Returns
        -------
        x : numpy array
            The parameters in fitting space.
        """
        x = np.zeros(len(p))
        for i in range(len(x)):
            x[i] = np.arcsin(np.sqrt((p[i] - lb[i]) / (ub[i] - lb[i])))
        return x

    # noinspection PyShadowingNames
    def model_params(x):
        """
        Transform from fitting space to model solve space.
        Parameters
        ----------
        x : numpy array
            The parameters in fitting space.

        Returns
        -------
        p : numpy array
            The parameters in model solve space.
        """
        p = np.zeros(len(x))
        for i in range(len(p)):
            p[i] = lb[i] + (ub[i] - lb[i]) * np.sin(x[i]) ** 2
        return p

    signed_error = ionbench.utils.cache.get_cached_signed_error(bm)
    grad = ionbench.utils.cache.get_cached_grad(bm, returnCost=True, residuals=True)

    # sample initial point
    if x0 is None:
        x0 = bm.sample()
    # Transform to fitting space
    x0 = fitting_params(x0)

    # Initialise variables
    weightedSSE = 0  # Will store the weighted SSE from the start of a reweighting cycle
    setInitialWeights = True  # Ensures the initial weights wInit are set at the start and then not overridden
    reweightMaxIter = 10  # Maximum number of times to reweight (update the weights) before resetting them to their initial values
    weightCounter = 0  # Number of times the weights have been updated
    iterCounter = 0  # Total number of curvilinear gradient descent iterations, compared against maxIter every time a local minimum is found to check for termination
    bestParams = x0  # Best params found so far
    bestCost = np.inf  # Best unweighted cost found so far
    wInit = None  # Initial weights
    w = None  # Current weights
    while True:
        # Initialisation routine for every weight update/reset
        falphaOld = np.inf  # Stores the weighted SSE from the previous iteration in curvilinear GD, for termination criteria of curvilinear gradient descent
        # find residuals and jacobian. Residuals will be used to find weights, J will be used in first iteration of curvilinear GD
        r0, J = grad(model_params(x0))
        currentCost = np.sqrt(np.mean(r0 ** 2))
        if debug:
            print(f'Starting out optimisation loop. Weight counter {weightCounter}')

        # initial weights at the start of optimisation. Only run once.
        if setInitialWeights:
            wInit = 10 / (10 + np.abs(r0))
            setInitialWeights = False  # Ensure never run again.
            if debug:
                print('Setting initial weights')
                print('Initial weights set at:')
                print(wInit)
                print(f'Last residual was {np.abs(r0[-1])} which gave weight {wInit[-1]}')

        if weightCounter == 0:
            # If start of reweighting cycle, set initial weights
            w = wInit
            # Set initial weighted SSE
            weightedSSE = np.dot(w @ r0, w @ r0)
            if debug:
                print('Setting weights at initial values')
                print(f'Setting initial weighted SSE at {weightedSSE}')
        elif weightCounter == 1 and currentCost > bestCost:
            # Perturb the best parameters and make that the starting location. Perturbation chosen to match http://dx.doi.org/https://doi.org/10.26190/unsworks/21047
            if debug:
                print(
                    f'Current cost ({currentCost}) was actually worse than best so far ({bestCost}) so randomly perturbing parameters which gave best cost')
            x0 = bestParams * np.random.uniform(low=0.9, high=1.1, size=bm.n_parameters())
        elif np.dot(r0, r0) < 0.3 * weightedSSE:
            # Weighted SSE has decreased by 70% so reset weights
            if debug:
                print(
                    f'Weighted SSE significantly decreased. Original value was {weightedSSE} and it is now {np.dot(r0, r0)} after {weightCounter} new weights')
            weightCounter = 0
            continue  # Restart weights
        elif weightCounter >= reweightMaxIter:
            # Maximum number of reweights has been applied so reset weights
            if debug:
                print(
                    f'Maximum number of weight updates. Original weighted SSE was {weightedSSE} and it is now {np.dot(r0, r0)} after {weightCounter} new weights')
            weightCounter = 0
            continue  # Restart weights
        else:
            # Update the weights
            wTmp = np.copy(w)
            w = weight_update(w, r0)
            if debug:
                print(f'Applied weight update. First weight used to be {wTmp[0]}, and is now {w[0]}')

        if currentCost < bestCost:
            # Update which is the best point so far
            bestParams = x0
            bestCost = currentCost

        # Begin curvilinear gradient descent
        for k in range(maxInnerIter):
            # Update number of curvilinear GD iterations
            iterCounter += 1
            if debug:
                print(f'Curvilinear GD Iteration: {k} of {maxInnerIter}')
            if k > 0:
                # If first iteration of GD, r0 and J have already been calculated
                r0, J = grad(model_params(x0))

            # apply weights to residuals and jacobian
            r0 = np.diag(w) @ r0
            J = np.diag(w) @ J

            # transform jacobian into fitting space
            for i in range(bm.n_parameters()):
                J[:, i] = J[:, i] * (ub[i] - lb[i]) * 2 * np.sin(x0[i]) * np.cos(x0[i])

            # a=2*J^T*r0
            a = 2 * np.transpose(J) @ r0
            # H = 2*J^T*J
            H = 2 * np.transpose(J) @ J

            # Eigen value decomposition on hessian H
            try:  # pragma: no cover
                d, V = np.linalg.eig(H)
                invV = np.linalg.inv(V)
            except np.linalg.LinAlgError as e:
                print('Encountered a fatal error in optimisation. Infs or nans in Hessian. Terminating early')
                x0 = model_params(x0)
                bm.evaluate()
                return x0
            if debug:
                print('Eigenvalues:')
                print(d)

            # Generate function of the curved path from starting point (no step, L(0)=[0,0,0,...]), initially moving in the steepest descent direction, converging towards Gauss Newton step at alpha=inf
            # noinspection PyPep8Naming,PyShadowingNames
            def L(alpha):
                m = np.zeros(len(d))
                for i in range(len(d)):
                    if d[i] == 0:  # pragma: no cover
                        m[i] = -alpha
                    else:  # pragma: no cover
                        m[i] = (np.exp(-2 * d[i] * alpha) - 1) / (2 * d[i])
                M = np.diag(m)
                return V @ M @ invV @ a

            if debug:
                print(f'L(0): {L(0)}, L(inf): {L(np.inf)}')

            # Minimise bm.SSE(x0+L(alpha)) with respect to alpha using Brent's method
            # Define function to find weighted SSE from alpha
            # Cache SSE function to help make the code to handle the brent method more robust without needing repeated function evaluations
            # noinspection PyPep8Naming,PyShadowingNames
            def SSE(alpha):
                weightedr0 = np.diag(w) @ signed_error(model_params(x0 + L(alpha)))
                return np.dot(weightedr0, weightedr0)

            # Run Brent's method to optimise SSE with respect to alpha
            try:  # pragma: no cover
                out = scipy.optimize.brent(SSE, brack=(0, 1, 1e9), tol=1e-8, full_output=True)
                alpha = out[0]
                falphaNew = out[1]
            except ValueError as e:  # pragma: no cover
                # Failed. Time to find out why and try to recover
                # Option 1: NaNs in either SSE(0) or SSE(1e9) but not SSE(1) so step still viable
                if (np.isnan(SSE(0)) or np.isnan(SSE(1e9))) and not np.isnan(SSE(1)):
                    # If SSE(1) is NOT WORSE than SSE(0) or SSE(1e9) (accounting for SSE(0) or SSE(1e9) being NaN), then we can assume that it is a good step
                    if debug:
                        print(
                            f'NaNs were found. SSE(0): {SSE(0)}, SSE(1): {SSE(1)}, SSE(1e9): {SSE(1e9)}. Will try to continue if possible')
                    if not SSE(1) > SSE(0) and not SSE(1) > SSE(1e9):
                        alpha = 1
                        falphaNew = SSE(alpha)
                    elif not SSE(1e-6) > SSE(0) and not SSE(1e-6) > SSE(1e9):
                        alpha = 1e-6
                        falphaNew = SSE(alpha)
                    elif not SSE(1e-9) > SSE(0) and not SSE(1e-9) > SSE(1e9):
                        alpha = 1e-9
                        falphaNew = SSE(alpha)
                    else:
                        print(f'Failed. SSE(0): {SSE(0)}, SSE(1e-9): {SSE(1e-9)}, SSE(1e-6): {SSE(1e-6)}, SSE(1): {SSE(1)}, SSE(1e9): {SSE(1e9)}')
                        x0 = model_params(x0)
                        bm.evaluate()
                        return x0
                # Option 2: SSE(1e9) is best - take full step
                elif SSE(1e9) <= SSE(1) and SSE(1e9) <= SSE(0):
                    # SSE(1e9)<SSE(1),SSE(0)
                    alpha = 1e9
                    falphaNew = SSE(alpha)
                    if debug:
                        print('Alpha was minimised (at least approximately) by taking the full step')
                # Option 3: SSE(0) is smallest - start with even smaller step than 1
                elif SSE(0) <= SSE(1):
                    # SSE(0)<SSE(1),SSE(1e9)
                    if debug:
                        print(
                            f'SSE(0) {SSE(0)} is smaller than SSE(1) {SSE(1)}, trying smaller steps for the initial guesses (1e-6 and 1e-9). If not, will assume converged')
                    if SSE(1e-6) < SSE(0):
                        initStep = 1e-6
                    elif SSE(1e-9) < SSE(0):
                        initStep = 1e-9
                    else:
                        # The step needed would be too small so assume converged
                        x0 = model_params(x0)
                        bm.evaluate()
                        return x0
                    out = scipy.optimize.brent(SSE, brack=(0, initStep, 1e9), tol=1e-8, full_output=True)
                    alpha = out[0]
                    falphaNew = out[1]
                else:
                    print(f"Failed to find bounds for Brent's method. SSE(0): {SSE(0)}, SSE(1): {SSE(1)}, SSE(1e9): {SSE(1e9)}")
                    x0 = model_params(x0)
                    bm.evaluate()
                    return x0

            # Take new step
            x0 = x0 + L(alpha)

            if debug:
                print(f'Optimal alpha was {alpha}')
                print(f'Resulting step is {L(alpha)}')
                print(f'falphaOld: {falphaOld}, falphaNew: {falphaNew}')

            if falphaNew <= costThreshold:  # pragma: no cover
                # If cost is below threshold, fully terminate the optimisation
                if debug:
                    print(f'Cost below threshold {costThreshold} found. falphaOld: {falphaOld}, falphaNew: {falphaNew}')
                x0 = model_params(x0)
                bm.evaluate()
                return x0

            # Check for local minimum/slow convergence. From Dokos 2003.
            if np.abs(falphaNew - falphaOld) <= 0.0001 * falphaNew:
                if debug:
                    print(f'Local minimum found, {k} of {maxInnerIter}')
                break

            falphaOld = falphaNew

            if bm.is_converged():
                break

        weightCounter += 1
        if debug:
            print(f'Weight counter incremented to {weightCounter}')
        if iterCounter >= maxIter:  # pragma: no cover
            if debug:
                print(
                    f'Reached maximum number of iterations so terminating early. iterCounter: {iterCounter}, maxIter: {maxIter}')
            bm.set_max_iter_flag()
            break

        if bm.is_converged():
            break

    x0 = model_params(x0)
    bm.evaluate()
    return x0


# noinspection PyShadowingNames
def get_modification(modNum=1):
    """
    modNum = 1 -> Dokos2004
    modNum = 2 -> Abed2013
    modNum = 3 -> Guo2010

    Returns
    -------
    mod : modification
        Modification corresponding to inputted modNum. Default is modNum = 1, so Dokos2004.

    """

    if modNum == 1:
        mod = ionbench.modification.Dokos2004()
    elif modNum == 2:
        mod = ionbench.modification.Abed2013()
    elif modNum == 3:
        mod = ionbench.modification.Guo2010()
    else:
        mod = ionbench.modification.Empty(name='dokos_gd')
    return mod

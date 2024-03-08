import numpy as np
import ionbench
import itertools
import copy
import matplotlib.pyplot as plt
import pickle


def second_deriv(bm, i, j, step=1e-4, buffer=1e-4):
    """
    Calculate the second order partial derivative of the log likelihood (specifically the negative of the sum of squared errors) in parameters i and j.

    Parameters
    ----------
    bm : benchmarker
        A benchmarker to define the log likelihood.
    i : int
        Index of the first parameter for the derivative.
    j : int
        Index of the second parameter for the derivative.
    step : float, optional
        The finite difference step size. If the step size is deemed too small, it will increase by step until the points used by finite difference are at least buffer away from the maximum likelihood. The default is 1e-4.
    buffer : float, optional
        While the evaluated negative sum of squared errors is within buffer of that at the maximum likelihood, the step size will be increased. The value of buffer should be at least 30 times larger than the noise caused by the ODE solver. The default is 1e-4.

    Returns
    -------
    deriv : float
        The second derivative of the negative sum of squared errors.

    """
    if i == j:
        def f(x):
            p = bm.input_parameter_space(bm._TRUE_PARAMETERS)
            p[i] = x
            return sse(bm, p)
        # fxx = f(x+h,y)-2f(x,y)+f(x-h,y) / h^2
        # Find h st f(x+h,y) < f(x,y)-buffer and f(x-h,y) < f(x,y)-buffer
        h = step
        x = bm.input_parameter_space(bm._TRUE_PARAMETERS)[i]
        centre = f(x)
        up = f(x + x * h)
        down = f(x - x * h)
        while up >= centre - buffer or down >= centre - buffer:
            h += step
            up = f(x + x * h)
            down = f(x - x * h)
        return (up - 2 * centre + down) / ((x * h)**2)
    else:
        def f(x, y):
            p = bm.input_parameter_space(bm._TRUE_PARAMETERS)
            p[i] = x
            p[j] = y
            return sse(bm, p)
        # fxy = f(x+h,y+k)-f(x+h,y-k)-f(x-h,y+k)+f(x-h,y-k) / 4hk
        # Find h st all of f(x+h,y+h), f(x-h,y+h), f(x+h,y-h), f(x-h,y-h) < f(x,y)-buffer
        h = step
        x = bm.input_parameter_space(bm._TRUE_PARAMETERS)[i]
        y = bm.input_parameter_space(bm._TRUE_PARAMETERS)[j]
        centre = f(x, y)
        up_up = f(x + x * h, y + y * h)
        up_down = f(x + x * h, y - y * h)
        down_up = f(x - x * h, y + y * h)
        down_down = f(x - x * h, y - y * h)
        while up_up >= centre - buffer or up_down >= centre - buffer or down_up >= centre - buffer or down_down >= centre - buffer:
            h += step
            up_up = f(x + x * h, y + y * h)
            up_down = f(x + x * h, y - y * h)
            down_up = f(x - x * h, y + y * h)
            down_down = f(x - x * h, y - y * h)
        return (up_up - up_down - down_up + down_down) / (4 * x * h * y * h)


def sse(bm, params):
    """
    Negative sum of squared errors.

    Parameters
    ----------
    bm : benchmarker
        Defines the cost function, solver and data.
    params : list
        List of parameters at which to find the negative sum of squared errors.

    Returns
    -------
    sse : float
        Negative sum of squared errors.

    """
    cost = bm.cost(params)
    return -cost**2 * len(bm.DATA)  # Sum of squared errors


def run(bm, sigma=1, preoptimise=True, ftol=3e-6, step=1e-4, buffer=1e-4):
    """
    Calculate the Fisher's Information Matrix for a benchmarker problem. Also plots the eigenspectrum of the FIM.

    Parameters
    ----------
    bm : benchmarker
        A benchmarker of which to find the FIM matrix.
    sigma : float, optional
        The variance of the noise in the benchmarker's data. The default is 1.
    preoptimise : bool, optional
        Should the true parameters be preoptimised to ensure they are the MLE, useful due to biases from noise. If so, then Scipy's Nelder Mead will be run starting from the true parameters. The default is True.
    ftol : float, optional
        The cost function tolerance to use for the Nelder Mead preoptimisation. Should be slightly larger than the noise in the solver. The default is 3e-6.
    step : float, optional
        Used in finite differences. The finite difference step size. If the step size is deemed too small, it will increase by step until the points used by finite difference are at least buffer away from the maximum likelihood. The default is 1e-4.
    buffer : float, optional
        Used in finite differences. While the evaluated negative sum of squared errors is within buffer of that at the maximum likelihood, the step size will be increased. The value of buffer should be at least 30 times larger than the noise caused by the ODE solver. The default is 1e-4.

    Returns
    -------
    mat : numpy matrix
        The Fisher's Information Matrix.

    """
    mat = np.zeros((bm.n_parameters(), bm.n_parameters()))

    bm._useScaleFactors = True
    # Search from the true parameters to find the actual MLE
    if preoptimise:
        out = ionbench.optimisers.scipy_optimisers.nelderMead_scipy.run(bm, x0=bm.input_parameter_space(bm._TRUE_PARAMETERS), ftol=ftol)
        bm._TRUE_PARAMETERS = bm.original_parameter_space(out)
        print('Identified MLE at:')
        print(bm._TRUE_PARAMETERS)
        print('Storing in this benchmarker object as the true parameters')
    for i, j in itertools.combinations(range(bm.n_parameters()), 2):
        d = second_deriv(bm, i, j, step=step, buffer=buffer) * 1 / (2 * sigma ** 2)
        mat[i, j] = d
        mat[j, i] = d
    for i in range(bm.n_parameters()):
        mat[i, i] = second_deriv(bm, i, i, step=step, buffer=buffer) * 1 / (2 * sigma ** 2)
    eigs = np.linalg.eigvals(mat)
    plt.eventplot(-eigs, orientation='vertical')
    plt.yscale('log')
    plt.xticks(ticks=[])
    plt.title('FIM Eigenspectrum: ' + bm.NAME)
    data = (eigs, mat)
    with open(bm.NAME + '_fim.pickle', 'wb') as f:
        pickle.dump(data, f)
    return mat


def explore_solver_noise(bm):
    """
    Identify the scale of noise in the solver cost function.

    Parameters
    ----------
    bm : benchmarker
        Defines the cost function and model solver.

    Returns
    -------
    None.

    """
    bm._useScaleFactors = True
    p = bm.input_parameter_space(bm._TRUE_PARAMETERS)
    params = [copy.copy(p) for _ in range(100)]
    for i in range(100):
        params[i][0] = 0.9999999**i
    # print(params)
    sses = [sse(bm, a) for a in params]
    plt.figure()
    plt.plot(sses)
    plt.show()
    print('STD: ' + str(np.std(sses)))
    print('Range: ' + str(max(sses) - min(sses)))

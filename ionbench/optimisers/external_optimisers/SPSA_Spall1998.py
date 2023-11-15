import numpy as np
import ionbench


def run(bm, x0=[], a=None, A=None, alpha=0.602, maxIter=1000, debug=False):
    """
    Implementation of SPSA from Spall 1998. There are few details on implementation in Maryak 1998, the paper which uses SPSA. Maryak cites Spall 1992, however this paper does not give recommendations for hyper-parameters, in which case we use Spall 1998. The Maryak modifications do not use scaling factors or log transforms, which likely significantly hurts the SPSA algorithm which takes step sizes of identical magnitudes in all parameter directions.
    Parameters
    ----------
    bm : Benchmarker
        A benchmarker to evaluate the performance of the optimisation algorithm.
    x0 : list, optional
        Initial parameter guess. If x0=[] (the default), then the population will be sampled from the benchmarker problems .sample() method.
    a : float, optional
        Hyperparameter to control the step size, ak. If no value is specified, an 'optimal' value will be approximated by finding a value of a which will produce step sizes of 0.1*geometric mean of x0.
    A : int, optional
        Hyperparameter to control the step size, ak. A controls the rate at which the step size decreases, large values of A will ensure the step size is similar for a long time. If no value is specified, it will be set to 0.1*maxIter, rounded to the nearest integer.
    alpha : float, optional
        Hyperparameter to control the step size. If no value is specified, an 'optimal' value of 0.602 will be used, as described in Spall 1998.
    maxIter : int, optional
        Maximum number of iterations. The default is 1000.
    debug : bool, optional
        If True, debug information will be printed, reporting that status of the optimisation each generation. The default is False.

    Returns
    -------
    xbest : list
        The best parameters identified.

    """
    if len(x0) == 0:
        # sample initial point
        x0 = bm.sample()

    if A is None:
        A = np.round(maxIter * 0.1)
        if debug:
            print('No value of A was specified. "Optimal" value determined as: ' + str(A))

    if a is None:
        # Find a such that there will be approximately a 'percentChangeInMinParam'*min(|x0|) sized step taken by SPSA during the early iterations.
        percentChangeInMinParam = 0.1
        grad = bm.grad(x0, inInputSpace=True)
        perturbVector = (np.random.rand(bm.n_parameters()) < 0.5) * 2 - 1
        approxGrad = np.dot(grad, perturbVector)
        a = percentChangeInMinParam * np.min(np.abs(x0)) * (A + 1)**alpha / np.abs(approxGrad)
        if debug:
            print('No value of a was specified. "Optimal" value determined as: ' + str(a))

    for k in range(maxIter):
        if debug:
            print('Iteration: ' + str(k) + ' of ' + str(maxIter))
        # Step size
        ak = a / (k + A + 1)**alpha
        grad = bm.grad(x0, inInputSpace=True)
        # Bernoulli (+-1) distributed random perturbation vector
        perturbVector = (np.random.rand(bm.n_parameters()) < 0.5) * 2 - 1
        # Gradient in random direction
        approxGrad = np.dot(grad, perturbVector)
        if debug:
            print('Old x0')
            print(x0)
            print('Others: grad, perturbVector, approxGrad, step size')
            print(grad)
            print(perturbVector)
            print(approxGrad)
            print(ak)
        x0 = x0 - ak * approxGrad * perturbVector
        if debug:
            print('New x0')
            print(x0)

    # Return the best point in the final simplex
    bm.evaluate(x0)
    return x0


if __name__ == '__main__':
    bm = ionbench.problems.staircase.HH_Benchmarker(sensitivities=True)
    x0 = bm.sample()
    x = run(bm, x0=x0, maxIter=1000, debug=False)


def get_modification(modNum=1):
    """
    modNum = 1 -> Maryak1998

    Returns
    -------
    mod : modification
        Modification corresponding to inputted modNum. Default is modNum = 1, so Maryak1998.

    """
    mod = ionbench.modification.Maryak1998()
    return mod

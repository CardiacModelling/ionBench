from scipy.stats import binom
import numpy as np


def mue(successes):
    """
    Calculate the median-unbiased estimator of the success rate.
    Parameters
    ----------
    successes : list
        A list of booleans indicating whether each run was successful.
    Returns
    -------
    mue : float
        The median-unbiased estimator of the success rate.
    """
    n = len(successes)
    x = np.sum(successes)
    if x == n:
        return 0.5 * (1 + 0.5 ** (1 / n))
    elif x == 0:
        return 0.5 * (1 - 0.5 ** (1 / n))
    p = np.linspace(0, 1, 10000)
    pR = p[len(p) - 1 - np.argmax((binom.cdf(x, n, p) >= 0.5)[::-1])]
    pL = p[np.argmax(1 - binom.cdf(x, n, p) + binom.pmf(x, n, p) >= 0.5)]
    mue = 0.5 * (pR + pL)
    return mue


def expected_time(times, successes):
    """
    Calculate the expected time to solve a problem using the median-unbiased estimator of the success rate.
    Parameters
    ----------
    times : list
        A list of times to solve the problem.
    successes : list
        A list of booleans indicating whether each run was successful.

    Returns
    -------
    expectedTime : float
        The expected time to successfully solve the problem.
    """
    if np.all(successes) or not np.any(successes):
        Tsucc = np.mean(times)
        Tfail = np.mean(times)
    else:
        Tsucc = np.mean([times[i] for i in range(len(times)) if successes[i]])
        Tfail = np.mean([times[i] for i in range(len(times)) if not successes[i]]) if not np.all(successes) else 0
    success_rate = mue(successes)
    expectedTime = Tsucc + Tfail * (1 - success_rate) / success_rate
    return expectedTime

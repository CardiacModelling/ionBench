from scipy.stats import beta
import numpy as np


def bootstrap_success_rate(success):
    """
    Draw a smooth bootstrapped sample of the success rate based on the observed successes.
    Parameters
    ----------
    success : list
        A list of booleans indicating whether each run was successful.

    Returns
    -------
    successRate : float
        The success rate of the sample.
    """

    return beta.rvs(np.sum(success) + 0.5, len(success) - np.sum(success) + 0.5)


def bootstrap_ERT(successes, times):
    """
    Generate a bootstrap sample of ERT.
    Parameters
    ----------
    successes : list
        A list of booleans indicating whether each run was successful.
    times : np.array
        Vector of times to generate the bootstrap sample from.
    Returns
    -------
    u : np.array
        The bootstrap sample of ERT.
    """
    count = len(successes)
    mask = np.random.choice(count, count)
    successes = successes[mask]
    times = times[mask]
    ERT = expected_time(times, successes, bootstrap=True)
    return ERT


def expected_time(times, successes, bootstrap=False):
    """
    Calculate the expected time to solve a problem using the median-unbiased estimator of the success rate.
    Parameters
    ----------
    times : list
        A list of times to solve the problem.
    successes : list
        A list of booleans indicating whether each run was successful.
    bootstrap : bool, optional
        If True, then whenever successes contains only True or only False, the corresponding missing time information is taken as a random sample from the times. Otherwise, the average of all times is taken.

    Returns
    -------
    expectedTime : float
        The expected time to successfully solve the problem.
    """
    if np.all(successes) or not np.any(successes):
        if np.all(successes) and bootstrap:
            Tfail = np.random.choice(times)
        else:
            Tfail = np.mean(times)
        if not np.any(successes) and bootstrap:
            Tsucc = np.random.choice(times)
        else:
            Tsucc = np.mean(times)
    else:
        Tsucc = np.mean([times[i] for i in range(len(times)) if successes[i]])
        Tfail = np.mean([times[i] for i in range(len(times)) if not successes[i]])
    if bootstrap:
        success_rate = bootstrap_success_rate(successes)
    else:
        success_rate = np.mean(successes)
    if success_rate == 0:
        return np.inf
    expectedTime = Tsucc + Tfail * (1 - success_rate) / success_rate
    return expectedTime

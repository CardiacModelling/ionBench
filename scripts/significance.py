import numpy as np
import matplotlib.pyplot as plt
import pandas
from scipy.stats import binom
from scipy.stats import beta
from scipy.interpolate import BSpline


# Define functions
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
        return 0.5*(1+0.5**(1/n))
    elif x == 0:
        return 0.5*(1-0.5**(1/n))
    p = np.linspace(0, 1, 10000)
    pR = p[len(p)-1-np.argmax((binom.cdf(x, n, p) >= 0.5)[::-1])]
    pL = p[np.argmax(1 - binom.cdf(x, n, p) + binom.pmf(x, n, p) >= 0.5)]
    mue = 0.5*(pR + pL)
    return mue


def bootstrap_success_rate(m, n):
    """
    Generate a bootstrap sample.
    Parameters
    ----------
    m : float
        The MUE of the success rate.
    n : int
        The length of the bootstrap sample.
    Returns
    -------
    u : np.array
        The bootstrap sample of the success rate.
    """
    u = np.random.rand(n)
    x = np.zeros(n)
    for i in range(n):
        x[i] = 1-beta.cdf(1-m, 3*u[i], 3*(1-u[i]))
    xBar = np.mean(x)
    s = BSpline(t=[0, 0, 0, 0, 0.293, 0.498, 0.699, 1, 1, 1, 1], c=[0.005, -0.033, 0.159, 0.495, 0.835, 1.033, 0.996], k=3)
    return np.clip(float(s(xBar)), 0, 1)


def bootstrap_times(times):
    """
    Generate a bootstrap sample.
    Parameters
    ----------
    times : np.array
        Vector of times to generate the bootstrap sample from.
    Returns
    -------
    u : np.array
        The bootstrapped sample of times.
    """
    sample = np.random.choice(times, size=len(times))
    return np.mean(sample)


def bootstrap_ERT(m, successes, times):
    """
    Generate a bootstrap sample of ERT.
    Parameters
    ----------
    m : int
        The MUE of the success rate.
    successes : list
        A list of booleans indicating whether each run was successful.
    times : np.array
        Vector of times to generate the bootstrap sample from.
    Returns
    -------
    u : np.array
        The bootstrap sample of ERT.
    """
    rateSample = bootstrap_success_rate(m, len(successes))
    if np.all(successes) or not np.any(successes):
        timeSample = bootstrap_times(times)
        ERT = timeSample/rateSample
    else:
        sucTimeSample = bootstrap_times(times[successes])
        failTimeSample = bootstrap_times(times[~successes])
        ERT = sucTimeSample + failTimeSample*rateSample/(1-rateSample)
    return ERT


bmShortNames = ['hh', 'mm', 'ikr', 'ikur', 'ina']
dfs = []
bootstrapCount = 1000
for bmShortName in bmShortNames:
    print(bmShortName)
    significance = []
    bestSamples = np.zeros(bootstrapCount)
    df = pandas.read_csv(f'resultsFile-{bmShortName}.csv')
    df2 = pandas.read_csv(f'resultsSummary-{bmShortName}.csv')
    maxRuns = int(df2['Success Count'][0] + df2['Failure Count'][0])
    for app in range(len(df)):
        if np.isnan(df2['ERT - Evals'][app]):
            significance.append(0)
            continue
        successes = np.array([df[f'Run {i} - Successful'][app] for i in range(maxRuns)])
        times = np.array([df[f'Run {i} - Cost Evals'][app] + df[f'Run {i} - Grad Evals'][app]*df2['Time Ratio'][app] for i in range(maxRuns)])
        m = mue(successes)
        if app == 0:
            for b in range(bootstrapCount):
                bestSamples[b] = bootstrap_ERT(m, successes, times)
            print(f'Mean ERT: {np.mean(bestSamples)}, SE ERT: {np.std(bestSamples)}, Median ERT: {np.median(bestSamples)}')
            significance.append(np.nan)
        else:
            samples = np.zeros(bootstrapCount)
            for b in range(bootstrapCount):
                samples[b] = bootstrap_ERT(m, successes, times)
            A, B = np.meshgrid(bestSamples, samples)
            print(f'{df["Optimiser Name"][app]} - {df["Mod Name"][app]}')
            significance.append(np.mean(B < A))
            print(significance[-1])
    df2['Significance'] = significance
    df2.to_csv(f'resultsSummary-{bmShortName}.csv', index=False, na_rep='NA')

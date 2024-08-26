import numpy as np
import matplotlib.pyplot as plt
import pandas
from scipy.stats import beta
from ionbench.utils.results import mue

# Inverting curve
# Digitized from Wang and Hutson 2013 - Figure 1
xInvert = [0, 9.6070e-2, 1.2955e-1, 1.5502e-1, 1.7613e-1, 1.9578e-1, 2.1325e-1, 5.0218e-2, 2.2999e-1, 2.4672e-1, 2.6274e-1,
     2.7875e-1, 2.9549e-1, 3.0932e-1, 3.2387e-1, 3.3843e-1, 3.5371e-1, 3.6754e-1, 3.8210e-1, 3.9592e-1, 4.1121e-1,
     4.2358e-1, 4.3886e-1, 4.5269e-1, 4.6652e-1, 4.8035e-1, 4.9345e-1, 5.0801e-1, 5.2256e-1, 5.3493e-1, 5.4876e-1,
     5.6332e-1, 5.7715e-1, 5.9098e-1, 6.0480e-1, 6.2009e-1, 6.3464e-1, 6.4847e-1, 6.6303e-1, 6.7831e-1, 6.9287e-1,
     7.0815e-1, 7.2271e-1, 7.3799e-1, 7.5400e-1, 7.7074e-1, 7.8748e-1, 8.0713e-1, 8.2606e-1, 8.4862e-1, 8.7336e-1,
     9.0975e-1, 9.4178e-1, 1]
yInvert = [0, 1.0917e-2, 3.0568e-2, 5.1310e-2, 7.2052e-2, 9.1703e-2, 1.1135e-1, 1.0917e-3, 1.3100e-1, 1.5175e-1, 1.7140e-1,
     1.9214e-1, 2.1179e-1, 2.3253e-1, 2.5218e-1, 2.7183e-1, 2.9258e-1, 3.1114e-1, 3.3188e-1, 3.5262e-1, 3.7227e-1,
     3.9192e-1, 4.1266e-1, 4.3231e-1, 4.5197e-1, 4.7162e-1, 4.9236e-1, 5.1201e-1, 5.3166e-1, 5.5131e-1, 5.7205e-1,
     5.9170e-1, 6.1135e-1, 6.3210e-1, 6.5175e-1, 6.7140e-1, 6.9323e-1, 7.1179e-1, 7.3144e-1, 7.5109e-1, 7.7293e-1,
     7.9148e-1, 8.1114e-1, 8.3188e-1, 8.5153e-1, 8.7118e-1, 8.9192e-1, 9.1266e-1, 9.3122e-1, 9.5087e-1, 9.7271e-1,
     9.9236e-1, 9.9891e-1, 1]


# Define functions
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
        x[i] = 1 - beta.cdf(1 - m, 3 * u[i], 3 * (1 - u[i]))
    xBar = np.mean(x)
    return np.interp(xBar, xInvert, yInvert)


def bootstrap_times(times, mask):
    """
    Generate a bootstrap sample.
    Parameters
    ----------
    times : np.array
        Vector of times to generate the bootstrap sample from.
    mask : list
        A list of booleans indicating which runs to include in the sample.

    Returns
    -------
    u : np.array
        The bootstrapped sample of times.
    """
    if not np.any(mask):
        # If there weren't any successful/failed runs, draw a single sample
        sample = np.random.choice(times)
    else:
        count = np.sum(mask)
        sample = np.random.choice(times[mask], size=count)
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
    failTimeSample = bootstrap_times(times, ~successes)
    sucTimeSample = bootstrap_times(times, successes)
    ERT = sucTimeSample + failTimeSample * (1 - rateSample) / rateSample
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
        times = np.array(
            [df[f'Run {i} - Cost Evals'][app] + df[f'Run {i} - Grad Evals'][app] * df2['Time Ratio'][app] for i in
             range(maxRuns)])
        m = mue(successes)
        if app == 0:
            for b in range(bootstrapCount):
                bestSamples[b] = bootstrap_ERT(m, successes, times)
            print(
                f'Mean ERT: {np.mean(bestSamples)}, SE ERT: {np.std(bestSamples)}, Median ERT: {np.median(bestSamples)}')
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
    df2['Rerun'] = np.array(significance) > 0.05  # If 5% of samples are better than best, result isn't significant so needs to be rerun
    df2.to_csv(f'resultsSummary-{bmShortName}.csv', index=False, na_rep='NA')

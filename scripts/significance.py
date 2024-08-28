import numpy as np
import matplotlib.pyplot as plt
import pandas
from ionbench.utils.results import bootstrap_ERT


bmShortNames = ['hh', 'mm', 'ikr', 'ikur', 'ina']
dfs = []
bootstrapCount = 10000
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
        if app == 0:
            for b in range(bootstrapCount):
                bestSamples[b] = bootstrap_ERT(successes, times)
            print(
                f'Mean ERT: {np.mean(bestSamples)}, SE ERT: {np.std(bestSamples)}, Median ERT: {np.median(bestSamples)}')
            print(f'ERT estimate: {df2["ERT - Evals"][app]}')
            significance.append(np.nan)
        else:
            samples = np.zeros(bootstrapCount)
            for b in range(bootstrapCount):
                samples[b] = bootstrap_ERT(successes, times)
            A, B = np.meshgrid(bestSamples, samples)
            print(f'{df["Optimiser Name"][app]} - {df["Mod Name"][app]}')
            significance.append(np.mean(B < A))
            print(significance[-1])
    df2['Significance'] = significance
    df2['Rerun'] = np.array(significance) > 0.05  # If 5% of samples are better than best, result isn't significant so needs to be rerun
    df2.to_csv(f'resultsSummary-{bmShortName}.csv', index=False, na_rep='NA')

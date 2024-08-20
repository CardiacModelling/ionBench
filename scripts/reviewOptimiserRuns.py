import ionbench
import numpy as np
import importlib
import csv
import os
import pandas
import re
from ionbench.utils.results import mue, expected_time


bms = [ionbench.problems.staircase.HH(), ionbench.problems.staircase.MM(), ionbench.problems.loewe2016.IKr(), ionbench.problems.loewe2016.IKur(), ionbench.problems.moreno2016.INa()]

for bm in bms:
    bm.plotter = False
    # Find optimiser short name
    bmShortName = bm.NAME.split('.')[1].lower()

    # Find out how many runs were attempted
    maxRuns = 0
    for app in ionbench.APP_UNIQUE:
        i = 0
        try:
            while True:
                bm.tracker.load(f"{bmShortName}_{app['module']}modNum{app['modNum']}_run{i}.pickle")
                i += 1
        except FileNotFoundError as e:
            maxRuns = max(maxRuns, i)
    print(f"{maxRuns} runs were attempted.")

    allData = []
    # Loop through all unique approaches
    for app in ionbench.APP_UNIQUE:
        # Print the approach and modification
        print('---------------')
        optimiserName = app['module'].split('.')[-1]
        mod = importlib.import_module(app['module']).get_modification(app['modNum'])
        modName = mod.NAME
        # Output data
        data = {'Optimiser Name': optimiserName, 'Mod Name': modName}
        print(f'Collating results for approach: {optimiserName}, modification: {modName}')
        try:
            bm.tracker.load(f"{bmShortName}_{app['module']}modNum{app['modNum']}_run{maxRuns-1}.pickle")
        except FileNotFoundError as e:
            print('Not all tracking files were found. Filling data with nans.')
            # Not all tracking files were found
            allData.append(data)
            continue
        for runNum in range(maxRuns):
            # For each run, load the tracking file and extract the data
            bm.tracker.load(f"{bmShortName}_{app['module']}modNum{app['modNum']}_run{runNum}.pickle")
            #bm.evaluate()
            # Get data at convergence
            i = bm.tracker.when_converged(bm.COST_THRESHOLD)
            a, b = bm.tracker.total_solve_time(i)
            data[f'Run {runNum} - Cost Time'] = a
            data[f'Run {runNum} - Grad Time'] = b
            i = -1 if i is None else i
            try:
                data[f'Run {runNum} - Successful'] = bm.tracker.bestCosts[i] < bm.COST_THRESHOLD
            except IndexError:
                data[f'Run {runNum} - Successful'] = False
            try:
                data[f'Run {runNum} - Cost Evals'] = bm.tracker.costSolves[i]
            except IndexError:
                data[f'Run {runNum} - Cost Evals'] = 0
            try:
                data[f'Run {runNum} - Grad Evals'] = bm.tracker.gradSolves[i]
            except IndexError:
                data[f'Run {runNum} - Grad Evals'] = 0
            try:
                data[f'Run {runNum} - Cost'] = bm.tracker.bestCosts[i]
            except IndexError:
                data[f'Run {runNum} - Cost'] = np.inf
            bm.reset()
        # Calculate the success rate
        successOrFail = [data[f'Run {i} - Successful'] for i in range(maxRuns)]
        data['Success Rate - MLE'] = np.mean(successOrFail)
        data['Success Rate'] = mue(successOrFail)
        # If at least one run succeeded
        data['Tier'] = 1 if np.any(successOrFail) else 2
        costTime = [data[f'Run {i} - Cost Time'] for i in range(maxRuns)]
        costEvals = [data[f'Run {i} - Cost Evals'] for i in range(maxRuns)]
        gradTime = [data[f'Run {i} - Grad Time'] for i in range(maxRuns)]
        gradEvals = [data[f'Run {i} - Grad Evals'] for i in range(maxRuns)]
        # Calculate average time per successful and failed run for cost and grad
        data['ERT - Time'] = expected_time(costTime, successOrFail) + expected_time(gradTime, successOrFail)
        data['ERT - Cost Evals'] = expected_time(costEvals, successOrFail)
        data['ERT - Grad Evals'] = expected_time(gradEvals, successOrFail)
        if data['Tier'] == 1:
            print(f'There were successes. Success rate (MLE): {data["Success Rate - MLE"]}, Success rate (MUE): {data["Success Rate"]}')
        else:
            print(f'There were no successes.')
        data['Expected Cost'] = np.mean([data[f'Run {i} - Cost'] for i in range(maxRuns)])
        data['Success Count'] = np.sum([data[f'Run {i} - Successful'] for i in range(maxRuns)])
        data['Failure Count'] = maxRuns - data['Success Count']
        data['Average Runtime'] = np.mean([data[f'Run {i} - Cost Time'] + data[f'Run {i} - Grad Time'] for i in range(maxRuns)])
        allData.append(data)
    df = pandas.DataFrame.from_records(allData)
    df = df.sort_values(['Tier', 'ERT - Time', 'Average Runtime'])
    df.to_csv(f'resultsFile-{bmShortName}.csv', index=False, na_rep='NA')

    # Produce summary information
    costEvals = np.sum([df[f'Run {i} - Cost Evals'].sum() for i in range(maxRuns)])
    costTime = np.sum([df[f'Run {i} - Cost Time'].sum() for i in range(maxRuns)])
    gradEvals = np.sum([df[f'Run {i} - Grad Evals'].sum() for i in range(maxRuns)])
    gradTime = np.sum([df[f'Run {i} - Grad Time'].sum() for i in range(maxRuns)])
    gradToCost = (gradTime/gradEvals)/(costTime/costEvals)
    df['Time Ratio'] = gradToCost
    df.loc[df['Optimiser Name'] == 'SPSA_Spall1998', 'Time Ratio'] = 2
    df['ERT - Evals'] = df['ERT - Cost Evals'] + df['Time Ratio']*df['ERT - Grad Evals']
    summary = df[['Optimiser Name', 'Mod Name', 'Tier', 'Success Rate', 'ERT - Time', 'ERT - Evals', 'Success Count', 'Failure Count', 'Average Runtime', 'Time Ratio']]
    df = df.sort_values(['Tier', 'ERT - Evals', 'Average Runtime'])
    summary.to_csv(f'resultsSummary-{bmShortName}.csv', index=False, na_rep='NA')
    print(f'Average cost time: {costTime/costEvals}, Average grad time: {gradTime/gradEvals}, Ratio: {gradToCost}')

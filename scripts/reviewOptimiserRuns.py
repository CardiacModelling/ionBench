import ionbench
import numpy as np
import importlib
import csv
import os
import pandas
import re


def expected_time(times, successes):
    Tsucc = np.mean([times[i] for i in range(len(times)) if successes[i]])
    Tfail = np.mean([times[i] for i in range(len(times)) if not successes[i]]) if not np.all(successes) else 0
    expectedTime = Tsucc + Tfail * (1 - np.mean(successes)) / np.mean(successes)
    return expectedTime


bm = ionbench.problems.staircase.MM()
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

with open(os.path.join(os.getcwd(), f'resultsFile-{bmShortName}.csv'), 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    # Set titles
    titles = ['Optimiser Name', 'Mod Name']
    # Variable names for run specific data
    variables = ['Cost', 'Cost Evals', 'Grad Evals', 'Cost Time', 'Grad Time',
                 'Successful']
    # Add run specific titles
    for i in range(maxRuns):
        for j in variables:
            titles.append(f'Run {i} - {j}')
    # Add final summary data titles
    titles += ['Tier', 'Success Rate', 'Expected Time', 'Expected Cost']
    # Write the title row
    writer.writerow(titles)
    # Loop through all unique approaches
    for app in ionbench.APP_UNIQUE:
        # Print the approach and modification
        print('---------------')
        optimiserName = app['module'].split('.')[-1]
        mod = importlib.import_module(app['module']).get_modification(app['modNum'])
        modName = mod.NAME
        # Output data
        row = [optimiserName, mod.NAME]
        print(f'Collating results for approach: {optimiserName}, modification: {modName}')
        try:
            bm.tracker.load(f"{bmShortName}_{app['module']}modNum{app['modNum']}_run{maxRuns-1}.pickle")
        except FileNotFoundError as e:
            print('Not all tracking files were found. Filling data with nans.')
            # Not all tracking files were found, fill data with nans
            row += [np.nan] * (len(titles) - 2)
            writer.writerow(row)
            continue
        # Data to track for each run
        costs = []
        costEvals = []
        gradEvals = []
        costTime = []
        gradTime = []
        successOrFail = []
        for runNum in range(maxRuns):
            # For each run, load the tracking file and extract the data
            bm.tracker.load(f"{bmShortName}_{app['module']}modNum{app['modNum']}_run{runNum}.pickle")
            bm.evaluate()
            # Get data at convergence
            i = bm.tracker.when_converged(bm.COST_THRESHOLD)
            i = -1 if i is None else i
            try:
                costs.append(bm.tracker.bestCosts[i])
            except IndexError:
                costs.append(np.inf)
            a, b = bm.tracker.total_solve_time(i)
            costTime.append(a)
            gradTime.append(b)
            try:
                gradEvals.append(bm.tracker.gradSolves[i])
            except IndexError:
                gradEvals.append(np.nan)
            try:
                costEvals.append(bm.tracker.modelSolves[i])
            except IndexError:
                costEvals.append(np.nan)
            successOrFail.append(costs[runNum] < bm.COST_THRESHOLD)
            row += [costs[runNum], costEvals[runNum], gradEvals[runNum], costTime[runNum], gradTime[runNum],
                    successOrFail[runNum]]
            bm.reset()
        # Calculate the success rate
        successRate = np.mean(successOrFail)
        # Replace nans with 0 in times incase either cost or grad was unused
        costTime = np.array([0 if np.isnan(t) else t for t in costTime])
        gradTime = np.array([0 if np.isnan(t) else t for t in costTime])
        if successRate > 0:
            # If at least one run succeeded
            # Calculate average time per successful and failed run for cost and grad
            time = expected_time(costTime, successOrFail) + expected_time(gradTime, successOrFail)
            tier = 1
            expectedCost = np.nan
            print(f'There were successes. Success rate: {successRate}, Expected Time: {time}')
        else:
            # If all failed, report average time and cost
            tier = 2
            time = np.mean(costTime + gradTime)
            expectedCost = np.mean(costs)
            print(f'There were no successes. Expected cost: {expectedCost}, Expected Time: {time}')
        row += [tier, successRate, time, expectedCost]
        writer.writerow(row)


# %%
def upper_rep(match):
    """
    Return the upper case of the first group in the match object.
    """
    return match.group(1).upper()


def title_rep(match):
    """
    Return the title case of the first group in the match object.
    """
    return match.group(1).title()


def full_name(s):
    """
    Return the full name of the optimiser or modification.
    """
    # Put in spaces before capital letters, ignoring acronyms, capitalise first letter
    s2 = s[0].upper()
    for i in range(1, len(s) - 4):
        if s[i].isupper() and not s[i - 1].isupper():
            s2 += ' ' + s[i]
        else:
            s2 += s[i]
    s2 += s[-4:]
    # Remove _
    s = s2.replace('_', '')
    # Put in spaces before years
    s = re.sub(r'(\d{4})', r' \1', s)
    # Space before scipy of pints
    s = re.sub(r'(scipy|pints)', r' (\1)', s)

    # Make remaining acronyms upper case
    s = re.sub(r'(?i)^(cmaes|lm|slsqp|ppso|pso|snes|xnes)', upper_rep, s)
    # Capitalise first letter in scipy and pints
    s = re.sub(r'(scipy|pints)', title_rep, s)
    return s


df = pandas.read_csv(f'resultsFile-{bmShortName}.csv')
df = df.sort_values(['Tier', 'Expected Time'])
df.to_csv(f'resultsFile-{bmShortName}-sorted.csv', index=False, na_rep='NA')

header = ['Optimiser', 'Modification', 'Success Rate (%)', 'Expected Time (s)']
caption = f'Results for the successful approaches in ionBench on the {bm.NAME} problem. NaN is reserved for results that either cannot be completed in a reasonable amount of time or for optimisers that are not yet finished. Abbreviations: GA - Genetic Algorithm, PSO - Particle Swarm Optimisation, TRR - Trust Region Reflective, PPSO - Perturbed Particle Swarm Optimisation, NM - Nelder Mead, DE - Differential Evolution, GD - Gradient Descent, CMAES - Covariance Matrix Adaption Evolution Strategy, SLSQP - Sequential Least SQuares Programming, LM - Levenberg-Marquardt.'
label = f'tab:resultsSucc-{bmShortName}'

df2 = df[df['Tier'] == 1]
df2.to_latex(buf=f'results-{bmShortName}-latex-succ.txt',
             columns=['Optimiser Name', 'Mod Name', 'Success Rate', 'Expected Time'], header=header,
             index=False, float_format='%.2f', formatters={'Mod Name': lambda x: full_name(x),
                                                           'Optimiser Name': lambda x: full_name(x),
                                                           'Success Rate': lambda x: int(x*100)},
             column_format='llrrr', longtable=True, label=label, caption=caption)

header = ['Optimiser', 'Modification', 'Expected Cost', 'Expected Time (s)']
caption = f'Results for the failed approaches in ionBench on the {bm.NAME} problem. NaN is reserved for results that either cannot be completed in a reasonable amount of time or for optimisers that are not yet finished. Abbreviations: GA - Genetic Algorithm, PSO - Particle Swarm Optimisation, TRR - Trust Region Reflective, PPSO - Perturbed Particle Swarm Optimisation, NM - Nelder Mead, DE - Differential Evolution, GD - Gradient Descent, CMAES - Covariance Matrix Adaption Evolution Strategy, SLSQP - Sequential Least SQuares Programming, LM - Levenberg-Marquardt.'
label = f'tab:resultsFail-{bmShortName}'
df3 = df[df['Tier'] != 1]
df3.to_latex(buf=f'results-{bmShortName}-latex-fail.txt',
             columns=['Optimiser Name', 'Mod Name', 'Expected Cost', 'Expected Time'], header=header,
             float_format='%.2f', formatters={'Mod Name': lambda x: full_name(x), 'Expected Cost': lambda x: f'{x:.4f}',
                                              'Optimiser Name': lambda x: full_name(x)},
             index=False, column_format='llrrrr', longtable=True, label=label, caption=caption)

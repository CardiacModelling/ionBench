import ionbench
import numpy as np
import importlib
import csv
import os
import pandas
import re

bm = ionbench.problems.staircase.MM()
with open(os.path.join(os.getcwd(), 'resultsFile.csv'), 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    titles = ['Optimiser Name', 'Mod Num', 'Mod Name']
    variables = ['Conv Cost', 'Best Cost', 'Cost FE', 'Grad FE', 'Average Cost FE Time', 'Average Grad FE Time',
                 'Successful']
    for i in range(5):
        for j in variables:
            titles.append(f'Run {i} - {j}')
    titles += ['Success Rate', 'Tier', 'Success Rate', 'Expected Time (Sensitivities)',
               'Expected Time (Finite Difference)', 'Expected Cost']
    writer.writerow(titles)
    # Loop through all unique approaches
    for app in ionbench.APP_UNIQUE:
        # Data to track for each run
        costAtConv = []
        bestCost = []
        costEvalsAtConv = []
        gradEvalsAtConv = []
        costAverTime = []
        gradAverTime = []
        try:
            # Check if all tracking files exist
            bm.tracker.load(f"{app['module']}modNum{app['modNum']}_run4.pickle")
        except Exception as e:
            # If they don't then skip this approach
            print(e)
            print(f"Tracking of {app['module']} failed.")
            optimiserName = app['module'].split('.')[-1]
            modNum = app['modNum']
            mod = importlib.import_module(app['module']).get_modification(app['modNum'])
            row = [optimiserName, modNum, mod._name]
            row += [np.nan] * (len(titles) - 3)
            writer.writerow(row)
            continue
        for runNum in range(5):
            # For each run, load the tracking file and extract the data
            bm.tracker.load(f"{app['module']}modNum{app['modNum']}_run{runNum}.pickle")
            costAverTime.append(np.mean(bm.tracker.costTimes))
            gradAverTime.append(np.mean(bm.tracker.gradTimes))
            bestCost.append(bm.tracker.bestCost)
            # Get data at convergence
            i = bm.tracker.when_converged(bm.costThreshold)
            i = -1 if i is None else i
            costAtConv.append(bm.tracker.costs[i])
            gradEvalsAtConv.append(bm.tracker.gradSolves[i])
            costEvalsAtConv.append(bm.tracker.modelSolves[i])
            bm.reset()
        # Calculate the success rate
        successOrFail = np.array(bestCost) < bm.costThreshold
        successRate = np.mean(successOrFail)
        # Handle times if either grad or cost wasn't used
        if all(np.isnan(gradAverTime)):
            gradAverTime = np.zeros(len(gradAverTime))
        if all(np.isnan(costAverTime)):
            costAverTime = np.zeros(len(costAverTime))
        if successRate > 0:
            # If at least one run succeeded
            # Calculate average time per successful and failed run for cost and grad
            # Cost times
            Tsucc = np.mean([costEvalsAtConv[t] * costAverTime[t] for t in range(5) if successOrFail[t]])
            Tfail = np.mean([costEvalsAtConv[t] * costAverTime[t] for t in range(5) if
                             not successOrFail[t]]) if successRate < 1 else 0
            expectedCostTime = Tsucc + Tfail * (1 - successRate) / successRate
            # Grad times
            Tsucc = np.mean([gradEvalsAtConv[t] * gradAverTime[t] for t in range(5) if successOrFail[t]])
            Tfail = np.mean([gradEvalsAtConv[t] * gradAverTime[t] for t in range(5) if
                             not successOrFail[t]]) if successRate < 1 else 0
            expectedGradTime = Tsucc + Tfail * (1 - successRate) / successRate
            tier = 1
            expectedCost = np.nan
            timeSens = expectedCostTime + expectedGradTime
            if 'SPSA' in app['module']:
                fdGradCountLow = 2
                fdGradCountUp = 2
            else:
                fdGradCountLow = bm.n_parameters()
                fdGradCountUp = bm.n_parameters() + 1
            fdTimeSuccLow = np.mean(
                [(costEvalsAtConv[t] + gradEvalsAtConv[t] * fdGradCountLow) * costAverTime[t] for t in range(5) if
                 successOrFail[t]])
            fdTimeSuccUp = np.mean(
                [(costEvalsAtConv[t] + gradEvalsAtConv[t] * fdGradCountUp) * costAverTime[t] for t in range(5) if
                 successOrFail[t]])
            fdTimeFailLow = np.mean(
                [(costEvalsAtConv[t] + gradEvalsAtConv[t] * fdGradCountLow) * costAverTime[t] for t in range(5) if
                 not successOrFail[t]]) if successRate < 1 else 0
            fdTimeFailUp = np.mean(
                [(costEvalsAtConv[t] + gradEvalsAtConv[t] * fdGradCountUp) * costAverTime[t] for t in range(5) if
                 not successOrFail[t]]) if successRate < 1 else 0
            fdTimeLow = fdTimeSuccLow + fdTimeFailLow * (1 - successRate) / successRate
            fdTimeUp = fdTimeSuccUp + fdTimeFailUp * (1 - successRate) / successRate
            fdTime = (fdTimeLow, fdTimeUp)
        else:
            tier = 2
            timeSens = np.mean(np.array(costEvalsAtConv) * costAverTime + np.array(gradEvalsAtConv) * gradAverTime)
            expectedCost = np.mean(bestCost)
            expectedCostEvals = np.nan
            expectedGradEvals = np.nan
            if 'SPSA' in app['module']:
                fdTimeLow = np.mean((np.array(costEvalsAtConv) + np.array(gradEvalsAtConv) * 2) * costAverTime)
                fdTimeUp = fdTimeLow
            else:
                fdTimeLow = np.mean(
                    (np.array(costEvalsAtConv) + np.array(gradEvalsAtConv) * (bm.n_parameters())) * costAverTime)
                fdTimeUp = np.mean(
                    (np.array(costEvalsAtConv) + np.array(gradEvalsAtConv) * (bm.n_parameters() + 1)) * costAverTime)
            fdTime = (fdTimeLow, fdTimeUp)
        optimiserName = app['module'].split('.')[-1]
        modNum = app['modNum']
        mod = importlib.import_module(app['module']).get_modification(app['modNum'])
        row = [optimiserName, modNum, mod._name]
        for i in range(5):
            row += [costAtConv[i], bestCost[i], costEvalsAtConv[i], gradEvalsAtConv[i], costAverTime[i],
                    gradAverTime[i], successOrFail[i]]
        row += [successRate, tier, successRate, timeSens, fdTime, expectedCost]
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


df = pandas.read_csv('resultsFile.csv')
df = df.sort_values(['Tier', 'Expected Time (Sensitivities)'])
df.to_csv('resultsFile-sorted.csv', index=False, na_rep='NA')

header = ['Optimiser', 'Modification', 'Tier', 'Success Rate', 'Expected Time (Sensitivities; s)',
          'Expected Time (Finite Difference; s)']
caption = 'Preliminary results for the successful approaches in ionBench. NaN is reserved for results that either cannot be completed in a reasonable amount of time or for optimisers that are not yet finished. Abbreviations: GA - Genetic Algorithm, PSO - Particle Swarm Optimisation, TRR - Trust Region Reflective, PPSO - Perturbed Particle Swarm Optimisation, NM - Nelder Mead, DE - Differential Evolution, GD - Gradient Descent, CMAES - Covariance Matrix Adaption Evolution Strategy, SLSQP - Sequential Least SQuares Programming, LM - Levenberg-Marquardt.'
label = 'tab:prelimResultsSucc'

df2 = df[df['Tier'] == 1]
df2.to_latex(buf='results-latex-succ.txt',
             columns=['Optimiser Name', 'Mod Name', 'Tier', 'Success Rate', 'Expected Time (Sensitivities)',
                      'Expected Time (Finite Difference)'], header=header, index=False, float_format='%.2f',
             formatters={'Tier': lambda x: int(x), 'Optimiser Name': lambda x: full_name(x),
                         'Mod Name': lambda x: full_name(x),
                         'Expected Time (Finite Difference)': lambda x: f'({eval(x)[0]:.2f}, {eval(x)[1]:.2f})'},
             column_format='llrrr', longtable=True, label=label, caption=caption)

header = ['Optimiser', 'Modification', 'Tier', 'Success Rate', 'Time (Sensitivities; s)', 'Time (Finite Difference; s)',
          'Expected Cost']
caption = 'Preliminary results for the failed approaches in ionBench. NaN is reserved for results that either cannot be completed in a reasonable amount of time or for optimisers that are not yet finished. Abbreviations: GA - Genetic Algorithm, PSO - Particle Swarm Optimisation, TRR - Trust Region Reflective, PPSO - Perturbed Particle Swarm Optimisation, NM - Nelder Mead, DE - Differential Evolution, GD - Gradient Descent, CMAES - Covariance Matrix Adaption Evolution Strategy, SLSQP - Sequential Least SQuares Programming, LM - Levenberg-Marquardt.'
label = 'tab:prelimResultsFail'
df3 = df[df['Tier'] != 1]
df3.to_latex(buf='results-latex-fail.txt',
             columns=['Optimiser Name', 'Mod Name', 'Tier', 'Success Rate', 'Expected Time (Sensitivities)',
                      'Expected Time (Finite Difference)', 'Expected Cost'], header=header, index=False,
             float_format='%.2f', formatters={'Tier': lambda x: int(x), 'Optimiser Name': lambda x: full_name(x),
                                              'Mod Name': lambda x: full_name(x), 'Expected Cost': lambda x: f'{x:.4f}',
                                              'Expected Time (Finite Difference)': lambda
                                              x: f'({eval(x)[0]:.2f}, {eval(x)[1]:.2f})'}, column_format='llrrrr',
             longtable=True, label=label, caption=caption)

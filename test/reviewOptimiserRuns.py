import ionbench
import numpy as np
import importlib
import csv
import os
import pandas
import re

bm = ionbench.problems.staircase.HH()
with open(os.path.join(os.getcwd(), 'resultsFile.csv'), 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    titles = ['Optimiser Name', 'Mod Num', 'Mod Name']
    variables = ['Conv Cost', 'Best Cost', 'Cost FE', 'Grad FE', 'Parameters Identified', 'Average Cost FE Time',
                 'Average Grad FE Time']
    for i in range(5):
        for j in variables:
            titles.append(f'Run {i} - {j}')
    titles += ['Success Rate', 'Expected Cost FE', 'Expected Cost Time', 'Expected Grad FE', 'Expected Grad Time',
               'Tier', 'Tier Score (Sensitivities)', 'Tier Score (Finite Difference)', 'Expected Time (Sensitivities)',
               'Expected Time (Finite Difference)', 'Expected Cost']
    writer.writerow(titles)
    for app in ionbench.APP_UNIQUE:
        costAtConv = []
        bestCost = []
        costEvalsAtConv = []
        gradEvalsAtConv = []
        paramIdenAtConv = []
        costAverTime = []
        gradAverTime = []
        try:
            bm.tracker.load(f"{app['module']}modNum{app['modNum']}_run4.pickle")
        except Exception as e:
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
            bm.tracker.load(f"{app['module']}modNum{app['modNum']}_run{runNum}.pickle")
            costAverTime.append(np.mean(bm.tracker.costTimes))
            gradAverTime.append(np.mean(bm.tracker.gradTimes))
            bestCost.append(bm.tracker.bestCost)
            finalParamId = bm.tracker.paramIdentifiedCount[-1]
            paramIdenAtConv.append(finalParamId)
            ifEqualFinalParamId = bm.tracker.paramIdentifiedCount == finalParamId
            ind = [i for i, x in enumerate(ifEqualFinalParamId) if
                   x]  # Indexes where number of parameters identified is equal to the final count
            for i in ind:
                if all(ifEqualFinalParamId[i:]):
                    # All future points remain with this many parameters identified, therefore it is considered converged
                    costEvalsAtConv.append(bm.tracker.modelSolves[i])
                    gradEvalsAtConv.append(bm.tracker.gradSolves[i])
                    costAtConv.append(bm.tracker.costs[i])
                    break
            bm.reset()
        successRate = np.mean(np.array(paramIdenAtConv) == bm.n_parameters())
        if all(np.isnan(gradAverTime)):
            gradTime = 0
        else:
            gradTime = np.mean(gradAverTime, where=np.logical_not(np.isnan(gradAverTime)))
        if all(np.isnan(costAverTime)):
            costTime = 0
        else:
            costTime = np.mean(costAverTime, where=np.logical_not(np.isnan(costAverTime)))
        if successRate == 1:
            expectedCostEvals = np.mean(costEvalsAtConv)
            expectedGradEvals = np.mean(gradEvalsAtConv)
            tier = 1
            score = expectedCostEvals * costTime + expectedGradEvals * gradTime
            timeSens = np.nan
            timeFD = np.nan
            expectedCost = np.nan
            if 'SPSA' in app['module']:
                scoreFD = (expectedCostEvals + expectedGradEvals * 2) * costTime
            else:
                scoreFD = (expectedCostEvals + expectedGradEvals * (bm.n_parameters() + 1)) * costTime
        elif successRate > 0:
            Tsucc = np.mean([costEvalsAtConv[t] for t in range(5) if paramIdenAtConv[t] == bm.n_parameters()])
            Tfail = np.mean([costEvalsAtConv[t] for t in range(5) if paramIdenAtConv[t] != bm.n_parameters()])
            expectedCostEvals = Tsucc + Tfail * (1 - successRate) / successRate
            Tsucc = np.mean([gradEvalsAtConv[t] for t in range(5) if paramIdenAtConv[t] == bm.n_parameters()])
            Tfail = np.mean([gradEvalsAtConv[t] for t in range(5) if paramIdenAtConv[t] != bm.n_parameters()])
            expectedGradEvals = Tsucc + Tfail * (1 - successRate) / successRate
            tier = 2
            timeSens = np.nan
            timeFD = np.nan
            expectedCost = np.nan
            score = expectedCostEvals * costTime + expectedGradEvals * gradTime
            if 'SPSA' in app['module']:
                scoreFD = (expectedCostEvals + expectedGradEvals * 2) * costTime
            else:
                scoreFD = (expectedCostEvals + expectedGradEvals * (bm.n_parameters() + 1)) * costTime
        else:
            tier = 3
            timeSens = np.mean(np.array(costEvalsAtConv) * costTime + np.array(gradEvalsAtConv) * gradTime)
            expectedCost = np.mean(bestCost)
            expectedCostEvals = np.nan
            expectedGradEvals = np.nan
            score = np.nan
            scoreFD = np.nan
            if 'SPSA' in app['module']:
                timeFD = np.mean(np.array(costEvalsAtConv) + np.array(gradEvalsAtConv) * 2) * costTime
            else:
                timeFD = np.mean(np.array(costEvalsAtConv) + np.array(gradEvalsAtConv) * (
                        bm.n_parameters() + 1)) * costTime
        optimiserName = app['module'].split('.')[-1]
        modNum = app['modNum']
        mod = importlib.import_module(app['module']).get_modification(app['modNum'])
        row = [optimiserName, modNum, mod._name]
        for i in range(5):
            row += [costAtConv[i], bestCost[i], costEvalsAtConv[i], gradEvalsAtConv[i], paramIdenAtConv[i],
                    costAverTime[i], gradAverTime[i]]
        row += [successRate, expectedCostEvals, costTime, expectedGradEvals, gradTime, tier, score, scoreFD, timeSens,
                timeFD, expectedCost]
        writer.writerow(row)


# %%
def upper_rep(match):
    return match.group(1).upper()


def title_rep(match):
    return match.group(1).title()


def full_name(s):
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
df = df.sort_values(['Tier', 'Tier Score (Sensitivities)'])
df.to_csv('resultsFile-sorted.csv', index=False, na_rep='NA')

header = ['Optimiser', 'Modification', 'Tier', 'Tier Score (Sensitivities; s)', 'Tier Score (Finite Difference; s)']
caption = 'Preliminary results for the successful approaches in ionBench. NaN is reserved for results that either cannot be completed in a reasonable amount of time or for optimisers that are not yet finished. Abbreviations: GA - Genetic Algorithm, PSO - Particle Swarm Optimisation, TRR - Trust Region Reflective, PPSO - Perturbed Particle Swarm Optimisation, NM - Nelder Mead, DE - Differential Evolution, GD - Gradient Descent, CMAES - Covariance Matrix Adaption Evolution Strategy, SLSQP - Sequential Least SQuares Programming, LM - Levenberg-Marquardt.'
label = 'tab:prelimResultsSucc'

df.to_latex(buf='results-latex-succ.txt', columns=['Optimiser Name', 'Mod Name', 'Tier', 'Tier Score (Sensitivities)',
                                                   'Tier Score (Finite Difference)'], header=header, index=False,
            float_format='%.2f', formatters={'Tier': lambda x: int(x), 'Optimiser Name': lambda x: full_name(x),
                                             'Mod Name': lambda x: full_name(x)}, column_format='llrrr', longtable=True,
            label=label, caption=caption)

header = ['Optimiser', 'Modification', 'Tier', 'Time (Sensitivities; s)', 'Time (Finite Difference; s)',
          'Expected Cost']
caption = 'Preliminary results for the failed approaches in ionBench. NaN is reserved for results that either cannot be completed in a reasonable amount of time or for optimisers that are not yet finished. Abbreviations: GA - Genetic Algorithm, PSO - Particle Swarm Optimisation, TRR - Trust Region Reflective, PPSO - Perturbed Particle Swarm Optimisation, NM - Nelder Mead, DE - Differential Evolution, GD - Gradient Descent, CMAES - Covariance Matrix Adaption Evolution Strategy, SLSQP - Sequential Least SQuares Programming, LM - Levenberg-Marquardt.'
label = 'tab:prelimResultsFail'
df.to_latex(buf='results-latex-fail.txt',
            columns=['Optimiser Name', 'Mod Name', 'Tier', 'Expected Time (Sensitivities)',
                     'Expected Time (Finite Difference)', 'Expected Cost'], header=header, index=False,
            float_format='%.2f', formatters={'Tier': lambda x: int(x), 'Optimiser Name': lambda x: full_name(x),
                                             'Mod Name': lambda x: full_name(x)}, column_format='llrrrr',
            longtable=True,
            label=label, caption=caption)

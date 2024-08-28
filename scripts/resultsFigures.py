import numpy as np
import matplotlib.pyplot as plt
import pandas
import re
import os
import ionbench


def apply_identifiers(name, types):
    ids = 'abcdefghijklmnopqrstuvwxyz'
    for i, t in enumerate(types):
        if t in name:
            name += ids[i]
    return name


def sorting_score(x, xmin, xmax):
    return np.linalg.norm((x-xmin)/(xmax-xmin))


def simplify_name(name):
    # Apply any a-z identifiers
    if 'Balser' in name:
        name = apply_identifiers(name, ['lm', 'nelderMead'])
    if 'Vanier' in name:
        name = apply_identifiers(name, ['conjugate', 'SA', 'stochastic', 'random'])
    if 'Sachse' in name:
        name = apply_identifiers(name, ['conjugate', 'powell'])
    if 'Gurkiewicz' in name and 'Ben' not in name:
        name = apply_identifiers(name, ['a', 'b'])
    if 'Seemann' in name:
        name = apply_identifiers(name, ['pso', 'powell'])
    if 'Wilhelms' in name:
        # Wilhelms is already identified by modification
        pass
    if 'Loewe' in name:
        if name.startswith('PSOTRR'):
            name = 'extra_identifier' + name
        name = apply_identifiers(name, ['ZZZ', 'PSO_', 'extra_identifier', 'hybridPSOTRR_', 'hybridPSOTRRTRR_'])

    # Remove optimiser information
    name = name.split(' - ')[-1]

    # Special characters
    if 'Szmek' in name:
        name = name.replace('Jedrzej', 'Jȩdrzej')
        name = name.replace('Szmek', '-Szmek')
    if 'Orovio' in name:
        name = name.replace('Orovio', '-Orovio')
    if 'Shalom' in name:
        name = name.replace('Shalom', '-Shalom')

    return name


# noinspection PyShadowingNames
def success_plot(dfs, titles, supp_plot=False):
    """
    Plot the success rate of each optimiser.
    Parameters
    ----------
    dfs : list
        A list of pandas.DataFrames containing the optimiser run data to plot.
    titles : list
        A list of strings for the titles of the plots.
    supp_plot : bool
        Whether to plot the supplementary figure (separate with and without sensitivity solves for ERT).
    """
    # Create subplot figure
    fig, axs = plt.subplots(3, 2, figsize=(7.5, 8.5), layout='constrained')
    # Maximum number of successful approaches across the problems
    maxSuccess = np.max([len(df[df['Tier'] == 1]) for df in dfs])
    for i in range(len(dfs)):
        df = dfs[i]
        df = df[df['Tier'] == 1]
        title = titles[i]
        y = np.zeros(len(df))  # Expected time
        y1 = np.zeros(len(df))  # Expected time without sensitivity
        y2 = np.zeros(len(df))  # Expected time with sensitivity
        x = []  # Approach names
        for j in range(len(df)):
            if supp_plot:
                y1[j] = df['ERT - Cost Evals'][j]
                y2[j] = df['ERT - Grad Evals'][j]
            else:
                y[j] = df['ERT - Evals'][j]
            x.append(simplify_name(df['Optimiser Name'][j] + ' - ' + df['Mod Name'][j]))
        # Bar chart plot
        colours = ['#DBB40C'] + ['#1F77B4']*(len(df)-1)
        if supp_plot:
            axs[i // 2, i % 2].bar(np.arange(len(y1)), y1, log=True, zorder=3, color='#1f77b4', width=0.4)
            axs[i // 2, i % 2].bar(np.arange(len(y2)) + 0.4, y2, log=True, zorder=3, color='#ff7f0e', width=0.4)
            axs[i // 2, i % 2].set_xticks(np.arange(len(y)) + 0.2, x)
            axs[0, 1].legend(['Without Sensitivity', 'With Sensitivity'])
            axs[i // 2, i % 2].set_xlim(-0.6, maxSuccess)
            axs[i // 2, i % 2].set_ylim(10, 1e7)
        else:
            axs[i // 2, i % 2].bar(np.arange(len(y)), y, tick_label=x, log=True, zorder=3, color=colours)
            axs[i // 2, i % 2].set_xlim(-1, maxSuccess)
            axs[i // 2, i % 2].set_ylim(1e2, 1e7)
        # Rotate x-axis labels
        plt.setp(axs[i // 2, i % 2].get_xticklabels(), rotation=30, ha='right', rotation_mode='anchor')
        # Set title and ylabel
        axs[i // 2, i % 2].title.set_text(title)
        axs[i // 2, i % 2].set_ylabel('ERT (FEs)')
        # Add y-axis grid lines
        axs[i // 2, i % 2].yaxis.grid(True, zorder=0)
        axs[i // 2, i % 2].minorticks_off()
    # Remove sixth sub-figure
    axs[2, 1].remove()
    plt.savefig(os.path.join(ionbench.ROOT_DIR, '..', 'scripts', 'figures', f'expectedTime{"-supp" if supp_plot else ""}.png'), bbox_inches='tight', dpi=300)
    plt.show()


def fail_plot(dfs, dfsSumm, titles):
    # Create subplot figure
    fig, axs = plt.subplots(3, 1, figsize=(7.5, 8.75), constrained_layout=True)
    axs = axs.flatten()
    # Settings for each subplot
    kwargs = {'sortVar': [1, 1, 2], 'plotVar': ['cost', 'time', 'cost'], 'placeTicks': [False, True, True], 'ylim': [(1e-5, 10), (1e2, 1e6), (1e-5, 10)]}
    for plotNum in range(len(axs)):
        data = []
        maxs = []
        mins = []
        for i in range(len(dfs)):
            df = dfs[i]
            xName = []
            originalIndex = []
            xTime = []
            xCost = []
            # Get minimum costs for each approach
            for j in range(len(df)):
                costs = []
                times = []
                run = 0
                while True:
                    try:
                        costs.append(df[f'Run {run} - Cost'][j])
                        t = df[f'Run {run} - Cost Evals'][j] + df[f'Run {run} - Grad Evals'][j]*dfsSumm[i]['Time Ratio'][j]
                        times.append(t)
                    except KeyError:
                        break
                    run += 1
                ind = np.argmin(costs)
                xName.append(simplify_name(df['Optimiser Name'][j] + ' - ' + df['Mod Name'][j]))
                originalIndex.append(j)
                xTime.append(times[ind])
                xCost.append(costs[ind])
            maxs.append(np.nanmax(xCost if kwargs['sortVar'][plotNum] == 2 else xTime))
            mins.append(np.nanmin(xCost if kwargs['sortVar'][plotNum] == 2 else xTime))
            sortedData = sorted(zip(xName, xTime, xCost, originalIndex), key=lambda x: x[kwargs['sortVar'][plotNum]])
            data.append(sortedData)
        # Find some global sorting for all approaches
        bestFit = []
        # Let each problem vote for the best approach
        scores = {}
        for prob, i in enumerate(data):  # For each sorted data set (different problems)
            for j in i:
                if j[0] not in scores:
                    scores[j[0]] = 0
                inc = sorting_score(j[kwargs['sortVar'][plotNum]], mins[prob], maxs[prob])  # Lower score means better approach
                if np.isnan(inc):
                    scores[j[0]] += 1
                else:
                    scores[j[0]] += inc
        # Sort data by score
        for i in data:
            sortedData = sorted(i, key=lambda x: scores[x[0]])
            bestFit.append(sortedData)
        # Plot the curves in global sort order
        for i in range(len(bestFit)):
            data = bestFit[i]
            df = dfs[i]
            xName, xTime, xCost, originalIndex = zip(*data)
            xName, xTime, xCost, originalIndex = list(xName), list(xTime), list(xCost), list(originalIndex)
            x = list(range(len(xName)))
            # Remove excess points (remove nans, remove tier 1 if plotting cost)
            for j in reversed(range(len(xName))):
                if df['Tier'][originalIndex[j]] not in [2]:
                    del xName[j]
                    del xTime[j]
                    del xCost[j]
                    del x[j]
            # Plot data
            if kwargs['plotVar'][plotNum] == 'time':
                axs[plotNum].semilogy(x, xTime, 'o-', label=titles[i] if plotNum == 0 else None)
            else:
                axs[plotNum].semilogy(x, xCost, 'o-', label=titles[i] if plotNum == 0 else None)
        # Set axis labels and ticks
        x = range(34)
        xName, _, _, _ = zip(*data)  # Not sure if this is sorted okay
        if kwargs['placeTicks'][plotNum]:
            axs[plotNum].set_xticks(ticks=x, labels=xName, rotation=90, horizontalalignment="center")
        else:
            axs[plotNum].set_xticks(ticks=x, labels=['']*len(x))
        if kwargs['plotVar'][plotNum] == 'time':
            axs[plotNum].set_ylabel('Time of best run')
        else:
            axs[plotNum].set_ylabel('Cost of best run')
        axs[plotNum].grid(axis='x')
        axs[plotNum].set_xlim((-0.5, 33.5))
        axs[plotNum].set_ylim(kwargs['ylim'][plotNum])
    # Add cost thresholds
    for i in axs[[0, 2]]:
        i.axhline(0.01558, -1, 34, color='#1f77b4', linestyle='--', label='Staircase HH - Cost Threshold' if i == axs[0] else None, zorder=0)
        i.axhline(0.005767, -1, 34, color='#ff7f0e', linestyle='--', label='Staircase MM - Cost Threshold' if i == axs[0] else None, zorder=0)
    # Titles and legends
    axs[0].set_title('Approaches sorted by time')
    axs[2].set_title('Approaches sorted by cost')
    fig.legend(loc="outside lower center", ncol=3)
    # Figure labels (A, B, C)
    fig.text(0.01, 0.982, 'A', fontsize=12, fontweight='bold')
    fig.text(0.01, 0.812, 'B', fontsize=12, fontweight='bold')
    fig.text(0.01, 0.44, 'C', fontsize=12, fontweight='bold')
    # Align ylabels
    fig.align_ylabels()
    fig.show()
    fig.savefig(os.path.join(ionbench.ROOT_DIR, '..', 'scripts', 'figures', 'allApproaches.png'), bbox_inches='tight', dpi=300)


# noinspection PyShadowingNames
def time_plot(dfs, titles, solveType='Cost'):
    """
    Plot the total solve time for each approach (cost or grad selected with solveType) across all runs against the average number of cost evaluations.
    Parameters
    ----------
    dfs : list
        A list of pandas.DataFrame containing the optimiser run data to plot.
    titles : list
        A list of strings for the titles of the plots.
    solveType : str
        The solve type to plot ('Cost' or 'Grad').

    Returns
    -------
    None.
    """
    # Setup subplot figure
    fig, axs = plt.subplots(3, 2, figsize=(7.5, 7.5), layout='constrained')
    for i in range(len(dfs)):
        # Get problem specific data and axes
        df = dfs[i]
        title = titles[i]
        ax = axs[i // 2, i % 2]
        fevals = []
        times = []
        # Get total number of function evals and total time for each problem
        for row in range(len(df)):
            f = []
            t = []
            run = 0
            while True:
                try:
                    f.append(df[f'Run {run} - {solveType} Evals'][row])
                    t.append(df[f'Run {run} - {solveType} Time'][row])
                except KeyError:
                    break
                run += 1
            fevals.append(np.sum(f, where=~np.isnan(f)))
            times.append(np.sum(t, where=~np.isnan(t)))
        # Remove NaNs and zeros
        fevals = np.array(fevals)
        times = np.array(times)
        pointsToKeep = np.logical_and(~np.isnan(fevals+times), fevals != 0, times != 0)
        fevals = fevals[pointsToKeep]
        times = times[pointsToKeep]
        # Plot scatter plot
        ax.scatter(fevals, times)
        # Add line of best fit (fitted on log-log scale)
        coeff = np.polyfit(np.log10(fevals), np.log10(times), 1)
        x = np.linspace(np.min(fevals), np.max(fevals), 1000)
        y = [10**coeff[1]*x_**coeff[0] for x_ in x]
        ax.plot(x, y, color='red', zorder=0)
        # Set log-log scale
        ax.set_xscale('log')
        ax.set_yscale('log')
        # Set axis labels
        ax.set_xlabel(f'{solveType} Evaluations')
        ax.set_ylabel('Time (s)')
        # Calculate average solve time across all solves
        totTime = np.sum(times)
        totEvals = np.sum(fevals)
        print(
            f'Problem: {title}, Solve Type: {solveType}, Total solves: {totEvals}, Total time: {totTime}, Average time per solve: {totTime / totEvals}')
        ax.title.set_text(f'{title}. Avg. Time: {totTime/ totEvals:.2e}s')
    # Remove sixth sub-figure
    axs[2, 1].remove()
    # Save and show plot
    plt.savefig(
        os.path.join(ionbench.ROOT_DIR, '..', 'scripts', 'figures', f'{solveType.lower()}Time.png'),
        bbox_inches='tight', dpi=300)
    plt.show()


bmShortNames = ['hh', 'mm', 'ikr', 'ikur', 'ina']
titles = ['Staircase HH', 'Staircase MM', 'Loewe 2016 IKr', 'Loewe 2016 IKur', 'Moreno 2016 INa']
dfsSumm = []
dfsFull = []
for bmShortName in bmShortNames:
    dfsSumm.append(pandas.read_csv(f'resultsSummary-{bmShortName}.csv'))
    dfsFull.append(pandas.read_csv(f'resultsFile-{bmShortName}.csv'))
success_plot(dfsSumm, titles)
success_plot(dfsFull, titles, supp_plot=True)
fail_plot(dfsFull, dfsSumm, titles)
time_plot(dfsFull, titles)
time_plot(dfsFull, titles, solveType='Grad')

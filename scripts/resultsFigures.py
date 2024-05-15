import numpy as np
import matplotlib.pyplot as plt
import pandas
import re
import os
import ionbench
from adjustText import adjust_text


def simplify_name(name):
    # Replace shorthand names
    name = name.replace('_scipy', '')
    name = name.replace('_pints', '')
    name = name.replace('trustRegionReflective', 'TRR')
    name = name.replace('lm ', 'LM ')
    name = name.replace('ppso', 'PPSO')
    name = name.replace('pso', 'PSO')
    name = name.replace('nelderMead', 'Nelder Mead')
    name = name.replace('powell', 'Powell')
    name = name.replace('patternSearch', 'Pattern Search')
    name = name.replace('randomSearch', 'Random Search')
    name = name.replace('stochasticSearch', 'Stochastic Search')
    name = name.replace('curvilinearGD', 'Curvilinear GD')
    name = name.replace('slsqp', 'SLSQP')
    name = name.replace('conjugateGD', 'Conjugate GD')
    name = name.replace('cmaes', 'CMA-ES')
    name = name.replace('hybridPSOTRRTRR', 'Hybrid PSO/TRR+TRR')
    name = name.replace('hybridPSOTRR', 'Hybrid PSO/TRR')
    name = name.replace('PSOTRR', 'PSO+TRR')
    name = name.replace('_', ' ')
    # Remove duplicate words (when paper name is in both optimiser and modification)
    words = name.split(sep=' ')
    words.reverse()
    words = sorted(set(words), key=words.index)
    words.reverse()
    name = ' '.join(words)
    # Add hyphens in names
    words = name.split(sep=' ')
    words[-1] = re.sub(r'(\w)([A-Z])', r'\1-\2', words[-1])
    name = ' '.join(words)
    # Add space before years
    name = re.sub(r'(.*)(\d{4})', r'\1 \2', name)
    # Remove any spaces around \n
    name = name.replace(' \n', '\n')
    name = name.replace('\n ', '\n')
    return name


# noinspection PyShadowingNames
def success_plot(dfs, titles):
    """
    Plot the success rate of each optimiser.
    Parameters
    ----------
    dfs : list
        A list of pandas.DataFrames containing the optimiser run data to plot.
    titles : list
        A list of strings for the titles of the plots.
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
        x = []  # Approach names
        for j in range(len(df)):
            y[j] = df['ERT - Evals'][j]
            x.append(simplify_name(df['Optimiser Name'][j] + ' - ' + df['Mod Name'][j]))
        # Bar chart plot
        axs[i // 2, i % 2].bar(np.arange(len(y)), y, tick_label=x, log=True, zorder=3)
        # Rotate x-axis labels
        plt.setp(axs[i // 2, i % 2].get_xticklabels(), rotation=30, ha='right', rotation_mode='anchor')
        # Set y-axis limits
        tmp = axs[i // 2, i % 2].get_ylim()
        axs[i // 2, i % 2].set_ylim(0.1, 10 ** np.ceil(np.log10(tmp[1])))
        # Set x-axis limits
        axs[i // 2, i % 2].set_xlim(-1, maxSuccess)
        # Set title and ylabel
        axs[i // 2, i % 2].title.set_text(title)
        axs[i // 2, i % 2].set_ylabel('ERT (FEs)')
        # Add y-axis grid lines
        axs[i // 2, i % 2].yaxis.grid(True, zorder=0)
    # Remove sixth sub-figure
    axs[2, 1].remove()
    plt.savefig(os.path.join(ionbench.ROOT_DIR, '..', 'scripts', 'figures', 'expectedTime.png'), bbox_inches='tight', dpi=300)
    plt.show()


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
            for run in range(10):
                f.append(df[f'Run {run} - {solveType} Evals'][row])
                t.append(df[f'Run {run} - {solveType} Time'][row])
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
titles = ['Staircase - HH', 'Staircase - MM', 'Loewe 2016 - IKr', 'Loewe 2016 - IKur', 'Moreno 2016 - INa']
dfs = []
for bmShortName in bmShortNames:
    df = pandas.read_csv(f'resultsSummary-{bmShortName}.csv')
    dfs.append(df)
success_plot(dfs, titles)

dfs = []
for bmShortName in bmShortNames:
    dfs.append(pandas.read_csv(f'resultsFile-{bmShortName}.csv'))
time_plot(dfs, titles)
time_plot(dfs, titles, solveType='Grad')

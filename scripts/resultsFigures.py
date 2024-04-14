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
    name = name.replace('slsqp', 'SLSQP')
    name = name.replace('conjugateGD', 'Conjugate GD')
    name = name.replace('cmaes', 'CMA-ES')
    name = name.replace('hybridPSOTRRTRR', 'Hybrid PSO-TRR+TRR')
    name = name.replace('hybridPSOTRR', 'Hybrid PSO-TRR')
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
    fig, axs = plt.subplots(3, 2, figsize=(10, 10), layout='constrained')
    # Maximum number of successful approaches across the problems
    maxSuccess = np.max([len(df[df['Tier'] == 1]) for df in dfs])
    for i in range(len(dfs)):
        df = dfs[i]
        df = df[df['Tier'] == 1]
        title = titles[i]
        y = np.zeros(len(df))  # Expected time
        x = []  # Approach names
        for j in range(len(df)):
            y[j] = df['Expected Time'][j]
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
        axs[i // 2, i % 2].set_ylabel('Expected Time (s)')
        # Add y-axis grid lines
        axs[i // 2, i % 2].yaxis.grid(True, zorder=0)
    # Remove six subfigure
    axs[2, 1].remove()
    plt.savefig(os.path.join(ionbench.ROOT_DIR, '..', 'scripts', 'figures', 'expectedTime.png'), bbox_inches='tight')
    plt.show()


def fail_plot(dfs, titles):
    """
    Plot the expected time of failed optimisers.
    Parameters
    ----------
    dfs : list
        A list of pandas.DataFrame containing the optimiser run data to plot.
    titles : list
        A list of strings for the titles of the plots.
    """
    for i in range(len(dfs)):
        df = dfs[i]
        title = titles[i]
        # Find optimal time and cutoff time
        df2 = df[df['Tier'] == 1].reset_index()
        optimalTime = np.min(df2['Expected Time'])
        nRuns = 10
        maxSuccessRate = 1-0.05**(1/nRuns)
        cutoffTime = optimalTime*maxSuccessRate
        # Create figure
        plt.figure(figsize=(8, 8), layout='constrained', dpi=300)
        # Get data, dropping SPSA for now while we decide how to time it
        df = df[np.logical_and(df['Tier'] == 2, df['Mod Name'] != 'Maryak1998')].reset_index()
        y = np.zeros(len(df))  # Expected cost
        x = np.zeros(len(df))  # Expected time
        names = []  # Approach names
        for j in range(len(df)):
            y[j] = df['Expected Cost'][j]
            x[j] = df['Expected Time'][j]
            names.append(simplify_name(df['Optimiser Name'][j] + ' \n ' + df['Mod Name'][j]))
        # Scatter plot of cost vs time
        plt.scatter(x, y, zorder=3)
        # Cutoff-time line
        plt.axvline(cutoffTime, color='k', linestyle='--', zorder=0)
        # Log-log plot
        axs = plt.gca()
        axs.set_yscale('log')
        axs.set_xscale('log')
        # Round limits to nearest power of 10
        tmp = axs.get_xlim()
        axs.set_xlim(10 ** np.floor(np.log10(tmp[0])), 10 ** np.ceil(np.log10(tmp[1])))
        tmp = axs.get_ylim()
        axs.set_ylim(10 ** np.floor(np.log10(tmp[0])), 10 ** np.ceil(np.log10(tmp[1])))
        # Add text labels and automatically adjust positions
        texts = [plt.text(x[i], y[i], names[i], ha='center', va='center') for i in range(len(x))]
        adjust_text(texts, arrowprops={'arrowstyle': '->', 'color': 'red'}, force_explode=(0.2, 0.3), force_static=(0.2, 0.3), force_text=(0.3, 0.3))
        # Set title, ylabel and xlabel
        axs.title.set_text(title)
        axs.set_ylabel('Expected Cost')
        axs.set_xlabel('Expected Time (s)')
        # Add y-axis grid lines
        axs.yaxis.grid(True, zorder=0)
        # Save and show plot
        plt.savefig(os.path.join(ionbench.ROOT_DIR, '..', 'scripts', 'figures', f'failed-{title.replace(" ", "").lower()}.png'), bbox_inches='tight')
        plt.show()


bmShortNames = ['hh', 'mm', 'ikr', 'ikur', 'ina']
titles = ['Staircase - HH', 'Staircase - MM', 'Loewe 2016 - IKr', 'Loewe 2016 - IKur', 'Moreno 2016 - INa']
dfs = []
for bmShortName in bmShortNames:
    dfs.append(pandas.read_csv(f'resultsFile-{bmShortName}-sorted.csv'))
success_plot(dfs, titles)
fail_plot(dfs, titles)

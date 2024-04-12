import numpy as np
import matplotlib.pyplot as plt
import pandas
import re
import os
import ionbench


def simplify_name(name):
    # Replace shorthand names
    name = name.replace('_scipy', '')
    name = name.replace('_pints', '')
    name = name.replace('trustRegionReflective', 'TRR')
    name = name.replace('slsqp', 'SLSQP')
    name = name.replace('conjugateGD', 'Conjugate GD')
    name = name.replace('cmaes', 'CMA-ES')
    name = name.replace('hybridPSOTRRTRR', 'Hybrid PSO-TRR+TRR')
    name = name.replace('hybridPSOTRR', 'Hybrid PSO-TRR')
    name = name.replace('PSOTRR', 'PSO+TRR')
    name = name.replace('_', ' ')
    # Remove duplicate words (when paper name is in both optimiser and modification)
    words = name.split()
    words.reverse()
    words = sorted(set(words), key=words.index)
    words.reverse()
    name = ' '.join(words)
    # Add hyphens in names
    words = name.split()
    words[-1] = re.sub(r'(\w)([A-Z])', r'\1-\2', words[-1])
    name = ' '.join(words)
    # Add space before years
    name = re.sub(r'(.*)(\d{4})', r'\1 \2', name)
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


bmShortNames = ['hh', 'mm', 'ikr', 'ikur', 'ina']
titles = ['Staircase - HH', 'Staircase - MM', 'Loewe 2016 - IKr', 'Loewe 2016 - IKur', 'Moreno 2016 - INa']
dfs = []
for bmShortName in bmShortNames:
    dfs.append(pandas.read_csv(f'resultsFile-{bmShortName}-sorted.csv'))
success_plot(dfs, titles)

import ionbench
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def staircase_plots(bm, paramPairs, nPoints):
    def f(x, y, i, j):
        p = np.copy(bm._TRUE_PARAMETERS)
        p[i] = x
        p[j] = y
        return bm.parameter_penalty(p) + bm.rate_penalty(p)

    modelType = 'HH' if 'hh' in bm.NAME else 'MM'
    vecF = np.vectorize(f)
    for i, j in paramPairs:
        # Create a contour plot of the penalty function to show the bounds
        x = np.logspace(np.log10(bm._LOWER_BOUND[i] * 0.3), np.log10(bm._UPPER_BOUND[i] * 3), nPoints)
        y = np.linspace(-0.1, bm._UPPER_BOUND[j] * 1.2, nPoints)
        x, y = np.meshgrid(x, y)
        z = vecF(x, y, i, j)
        plt.contour(x, y, z, [1e5])
        plt.gca().set_xscale('log')
        plt.scatter(bm._TRUE_PARAMETERS[i], bm._TRUE_PARAMETERS[j], marker='*')
        plt.title(f'Rate bounds - {modelType} Staircase - ($p_{{{i}}}$,$p_{{{j}}}$)')
        plt.savefig(f'figures/rateBounds-{modelType}-{i}.png')
        plt.close()
        # Create a contour plot of the penalty function at multiple levels
        x = np.logspace(np.log10(bm._LOWER_BOUND[i] * 0.01), np.log10(bm._UPPER_BOUND[i] * 10), nPoints)
        y = np.linspace(-0.2, bm._UPPER_BOUND[j] * 1.75, nPoints)
        x, y = np.meshgrid(x, y)
        z = vecF(x, y, i, j)
        CS = plt.contour(x, y, z, [1e5, 2e5, 5e5, 1e6])
        plt.gca().set_xscale('log')
        plt.title(f'Penalty function - {modelType} Staircase - ($p_{{{i}}}$,$p_{{{j}}}$)')
        plt.clabel(CS, inline=1, fontsize=10)
        plt.scatter(bm._TRUE_PARAMETERS[i], bm._TRUE_PARAMETERS[j], marker='*')
        plt.savefig(f'figures/penalty-{modelType}-' + str(i) + '.png')
        plt.close()
        # Create a heatmap of the penalty function
        plt.pcolor(x, y, z, norm=mpl.colors.LogNorm(vmin=1e5, vmax=z.max()))
        plt.gca().set_xscale('log')
        plt.savefig(f'figures/penaltyHeatmap-{modelType}-' + str(i) + '.png')
        plt.close()
        print(f'Parameter pair {i},{j} complete')


bm = ionbench.problems.staircase.HH()
staircase_plots(bm, [(0, 1), (2, 3), (4, 5), (6, 7)], nPoints=1000)

bm = ionbench.problems.staircase.MM()
staircase_plots(bm, [(0, 1), (2, 14), (3, 4), (5, 6), (7, 8), (10, 11), (12, 13), (9, 14)], nPoints=1000)

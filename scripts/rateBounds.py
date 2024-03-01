import ionbench
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

nPoints = 1000

bm = ionbench.problems.staircase.HH()


# Convert 2D into parameter vector
# Where penalty equals exactly 1e4 gives bounds
def f(x, y, i, j):
    p = np.copy(bm.defaultParams)
    p[i] = x
    p[j] = y
    return bm.parameter_penalty(p) + bm.rate_penalty(p)


vecF = np.vectorize(f)
for i in [0, 2, 4, 6]:
    j = i + 1
    # Create a contour plot of the penalty function to show the bounds
    x = np.logspace(np.log10(bm.lbStandard[i] * 0.3), np.log10(bm.ubStandard[i] * 3), nPoints)
    y = np.linspace(-0.1, bm.ubStandard[j] * 1.2, nPoints)
    x, y = np.meshgrid(x, y)
    z = vecF(x, y, i, j)
    plt.contour(x, y, z, [1e4])
    plt.gca().set_xscale('log')
    plt.scatter(bm.defaultParams[i], bm.defaultParams[j], marker='*')
    plt.title(f'Rate bounds - HH Staircase - ($p_{i}$,$p_{j}$)')
    plt.savefig('figures/rateBounds-HH-' + str(i) + '.png')
    plt.close()
    # Create a contour plot of the penalty function at multiple levels
    x = np.logspace(np.log10(bm.lbStandard[i] * 0.01), np.log10(bm.ubStandard[i] * 10), nPoints)
    y = np.linspace(-0.2, bm.ubStandard[j] * 1.75, nPoints)
    x, y = np.meshgrid(x, y)
    z = vecF(x, y, i, j)
    CS = plt.contour(x, y, z, [1e4, 2e4, 3e4, 4e4, 5e4, 1e5])
    plt.gca().set_xscale('log')
    plt.title(f'Penalty function - HH Staircase - ($p_{i}$,$p_{j}$)')
    plt.clabel(CS, inline=1, fontsize=10)
    plt.scatter(bm.defaultParams[i], bm.defaultParams[j], marker='*')
    plt.savefig('figures/penalty-HH-' + str(i) + '.png')
    plt.close()
    # Create a heatmap of the penalty function
    plt.pcolor(x, y, z, norm=mpl.colors.LogNorm(vmin=1e4, vmax=z.max()))
    plt.gca().set_xscale('log')
    plt.savefig('figures/penaltyHeatmap-HH-' + str(i) + '.png')
    plt.close()
    print(f'Parameter pair {i},{j} complete')

import ionbench
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import matplotlib as mpl
import os


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
        print(f'Parameter pair {i},{j} complete')


def paper_plot():
    """
    Plot the figure for the rate bounds in the paper.
    """
    fig = plt.figure(figsize=(5, 6.5), constrained_layout=True)
    subfigs = fig.subfigures(2, 1)
    axTops = subfigs[0].subplots(1, 2)

    # Plot the rate bounds
    def subplot_2d(ax, bm, V):
        lb = bm._LOWER_BOUND
        ub = bm._UPPER_BOUND
        rate_min = bm.RATE_MIN
        rate_max = bm.RATE_MAX
        ax.set_xscale('log')
        ax.set_yscale('linear')
        ax.set_xlabel('$A$')
        ax.set_ylabel('$b$')
        ax.set_title(f'$Aexp({"-" if V<0 else ""}bV)$')
        # Calculate rate bounds
        x = np.logspace(np.log10(lb[0]), np.log10(ub[0]), 1000)
        yu = np.log(rate_max/x)/np.abs(V)
        yl = np.log(rate_min/x)/np.abs(V)
        # Shade parameter bounds and rate bounds regions
        ax.fill_between(x=x, y1=lb[1], y2=np.clip(yl, lb[1], ub[1]), color='tab:blue', alpha=0.2)
        pb = ax.fill_between(x=x, y1=yu, y2=ub[1], color='tab:blue', alpha=0.2, label='Parameter bounds')
        rb = ax.fill_between(x=x, y1=np.clip(yl, lb[1], ub[1]), y2=yu, color='tab:orange', alpha=0.2, label='Rate bounds')
        # Plot rate bounds
        ax.plot(x, yu, 'tab:orange', linewidth=1.5)
        # Remove points where y<0
        x, yl = zip(*[(x[i], yl[i]) for i in range(len(x)) if yl[i] >= 0])
        ax.plot(x, yl, 'tab:orange', linewidth=1.5)
        # Plot parameter bounds
        ax.axvline(lb[0], ymin=-1, color='tab:blue', linewidth=1.5)
        ax.axvline(ub[0], ymin=-1, color='tab:blue', linewidth=1.5)
        ax.axhline(lb[1], xmax=1e4, color='tab:blue', linewidth=1.5)
        ax.axhline(ub[1], xmax=1e4, color='tab:blue', linewidth=1.5)
        # Set plot parameters
        ax.set_ylim([-0.02, 0.42])
        ax.set_xlim([lb[0] * 0.3, ub[0] / 0.3])
        ax.set_yticks([lb[1], (lb[1]+ub[1])/2, ub[1]])
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.set_xticks(np.logspace(-7, 3, 11), labels=["$10^{-7}$", "", "", "", "", "$10^{-2}$", "", "", "", "", "$10^{3}$"])
        xticks = []
        for i in range(-7, 3):
            xticks.append(2 * 10 ** i)
            xticks.append(4 * 10 ** i)
            xticks.append(6 * 10 ** i)
            xticks.append(8 * 10 ** i)
        ax.set_xticks(xticks, minor=True)
        ax.xaxis.set_minor_formatter("")
        if V < 0:
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position('right')
        return pb, rb

    bm = ionbench.problems.staircase.HH()
    # bm.V_HIGH = 58.25
    subplot_2d(ax=axTops[0], bm=bm, V=bm.V_HIGH)
    pb, rb = subplot_2d(ax=axTops[1], bm=bm, V=bm.V_LOW)
    leg = subfigs[0].legend(handles=[pb, rb], loc='outside lower center')
    for lh in leg.legend_handles:
        lh.set_alpha(1)

    # Plot the penalty function
    axBot = subfigs[1].subplots(1, 2)
    ax1 = axBot[0]
    ax2 = axBot[1]
    pl = np.linspace(-1, 1, 1000)
    pu = np.logspace(3, 5, 1000)
    lb = 1e-3
    ub = 1e4

    @np.vectorize
    def pen(pi, lb, ub):
        out = 0
        if pi < lb:
            out += 1e5
            out += 1e5 * np.log(1 + np.abs(pi - lb))
        if pi > ub:
            out += 1e5
            out += 1e5 * np.log(1 + np.abs(pi - ub))
        return out

    ax1.plot(pl, pen(pl, lb, ub))
    ax1.set_xscale('linear')
    ax1.set_title('Lower bound')
    ax1.set_ylabel('Penalty')
    ax2.plot(pu, pen(pu, lb, ub))
    ax2.set_xscale('log')
    ax2.set_title('Upper bound')
    ax2.sharey(ax1)

    # split axes
    # hide the spines
    ax1.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    # Move the ticks to the right side
    ax2.yaxis.tick_right()

    # Add diagonal lines // to the split
    d = .015  # "/" size
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    ax1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax2.plot((-d, +d), (-d, +d), **kwargs)

    # Add common x axis label
    subfigs[1].supxlabel('Parameter/Rate')

    # Figure titles
    subfigs[0].suptitle('Rate bounds')
    subfigs[1].suptitle('Penalty function')
    subfigs[0].text(0.02, 0.95, 'A', fontsize=12, fontweight='bold')
    subfigs[1].text(0.02, 0.95, 'B', fontsize=12, fontweight='bold')
    plt.savefig(
        os.path.join(ionbench.ROOT_DIR, '..', 'scripts', 'figures', 'rateBounds.png'),
        bbox_inches='tight', dpi=300)
    plt.show()


paper_plot()

#bm = ionbench.problems.staircase.HH()
#staircase_plots(bm, [(0, 1), (2, 3), (4, 5), (6, 7)], nPoints=1000)

#bm = ionbench.problems.staircase.MM()
#staircase_plots(bm, [(0, 1), (2, 14), (3, 4), (5, 6), (7, 8), (10, 11), (12, 13), (9, 14)], nPoints=1000)

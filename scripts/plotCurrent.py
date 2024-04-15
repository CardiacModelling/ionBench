import ionbench
import numpy as np
import matplotlib.pyplot as plt
from ionbench.problems import *
import os


def plot_current(bm, title):
    # Plot the current and voltage for a given problem
    current = bm.DATA
    time = np.arange(0, bm.T_MAX, bm.TIMESTEP)
    fig, ax1 = plt.subplots(layout='constrained')
    ax2 = ax1.twinx()
    # Plot current
    ax1.plot(time, current, 'r-', zorder=3)
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Current', color='r')
    # Plot voltage
    p = bm.protocol()
    ax2.plot(time, p.value_at_times(time), 'b-', zorder=1)
    ax2.set_ylabel('Voltage (mV)', color='b')
    # Line up zero current with reversal potential
    yC = ax1.get_ylim()
    yV = ax2.get_ylim()
    reversal = bm._ANALYTICAL_MODEL.default_membrane_potential()
    ax1.set_ylim([min(yC[0], -yC[1]), max(yC[1], -yC[0])])
    ax2.set_ylim([reversal - (yV[1] - reversal), yV[1]])
    ax1.set_xlim([0, bm.T_MAX])
    ax1.axhline(0, color='black', zorder=0)
    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.set_frame_on(False)
    ax1.set_title(title)
    # Save figure
    title = title.replace('\n', '')
    fig.savefig(os.path.join(ionbench.ROOT_DIR, '..', 'scripts', 'figures', f'data-{title.replace(" ", "").lower()}' + '.png'), bbox_inches='tight', dpi=300)
    plt.show()


def plot_moreno():
    # Plot moreno data
    bm = moreno2016.INa()
    data = bm.DATA
    # Split data into each summary statistic
    lengths = [9, 20, 10, 9]
    fig, ax1 = plt.subplots(layout='constrained')
    for i in range(len(lengths)):
        indexes = np.arange(sum(lengths[:i]), sum(lengths[:i+1]))
        current = data[indexes]
        ax1.plot(indexes, current, 'k')
        if i != len(lengths) - 1:
            ax1.axvline(sum(lengths[:i+1])-0.5, color='r')
    ax1.set_ylabel('Summary Statistics')
    ax1.set_xticks([])
    ax1.set_title('Moreno 2016\nINa Data')
    # Save figure
    fig.savefig(os.path.join(ionbench.ROOT_DIR, '..', 'scripts', 'figures', 'data-moreno2016inadata.png'), bbox_inches='tight', dpi=300)
    plt.show()


bms = [staircase.HH(), staircase.MM(), loewe2016.IKr(), loewe2016.IKur()]
titles = ['Staircase\nHH Data', 'Staircase\nMM Data', 'Loewe 2016\nIKr Data', 'Loewe 2016\nIKur Data']
for bm, title in zip(bms, titles):
    plot_current(bm, title)
plot_moreno()

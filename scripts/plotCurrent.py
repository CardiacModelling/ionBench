import ionbench
import numpy as np
import matplotlib.pyplot as plt
from ionbench.problems import *
import os


def plot_current(bm, title, ax1):
    if 'hh' in bm.NAME or 'ikr' in bm.NAME:
        left = True
    else:
        left = False
    # Plot the current and voltage for a given problem
    current = bm.DATA
    time = np.arange(0, bm.T_MAX, bm.TIMESTEP)
    ax2 = ax1.twinx()
    # Plot current
    ax1.plot(time, current, 'r-', zorder=3)
    ax1.set_xlabel('Time (ms)')
    if left:
        ax1.set_ylabel('Current', color='r', loc='center', labelpad=-4)
    else:
        ax1.set_ylabel('Current', color='r', loc='bottom', labelpad=0)
    # Plot voltage
    p = bm.protocol()
    ax2.plot(time, p.value_at_times(time), 'b-', zorder=1)
    if left:
        ax2.set_ylabel('Voltage (mV)', color='b', loc='top', labelpad=0)
    else:
        ax2.set_ylabel('Voltage (mV)', color='b', loc='center')
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


def plot_moreno(axs, xLabels, xCoords, titles, yLabels, xTicks):
    # Plot moreno data
    bm = moreno2016.INa()
    data = bm.DATA
    # Split data into each summary statistic
    lengths = [9, 20, 10, 9]
    for i in range(len(lengths)):
        indexes = np.arange(sum(lengths[:i]), sum(lengths[:i+1]))
        current = data[indexes]
        axs[f'm{i}'].plot(xCoords[i], current, 'r.-')
        if i == 2:
            axs[f'm{i}'].set_xscale('log')
        axs[f'm{i}'].set_xlabel(xLabels[i])
        axs[f'm{i}'].set_ylabel(yLabels[i])
        axs[f'm{i}'].set_title(titles[i])
        axs[f'm{i}'].set_xticks(xTicks[i])
        axs[f'm{i}'].grid(axis='y')


bms = [staircase.HH(), staircase.MM(), loewe2016.IKr(), loewe2016.IKur()]
titles = ['Staircase HH', 'Staircase MM', 'Loewe 2016 IKr', 'Loewe 2016 IKur']
panels = [[bms[0].NAME, bms[1].NAME,],
          [bms[2].NAME, bms[3].NAME]]
fig, axsDict = plt.subplot_mosaic(panels, figsize=(7.5, 5))
for bm, title in zip(bms, titles):
    plot_current(bm, title, axsDict[bm.NAME])
plt.subplots_adjust(left=0.09,
                    bottom=0.1,
                    right=0.9,
                    top=0.95,
                    wspace=0.55,
                    hspace=0.4)
fig.align_labels()
fig.savefig(
    os.path.join(ionbench.ROOT_DIR, '..', 'scripts', 'figures', f'data' + '.png'),
    bbox_inches='tight', dpi=300)
plt.show()

panels = [['m0', 'm1', 'm2', 'm3']]
fig, axsDict = plt.subplot_mosaic(panels, figsize=(7.5, 2.5), layout='constrained')
xLabels = ['Voltage (mV)', 'Voltage (mV)', 'Time (ms)', 'Voltage (mV)']
xCoords = [np.arange(-120, -30, 10), np.arange(-75, 25, 5),
           [1, 5, 10, 25, 50, 100, 150, 250, 500, 1000], np.arange(-20, 25, 5)]
xTicks = [np.arange(-120, -30, 40), np.arange(-75, 30, 50), [1, 10, 100, 1000], np.arange(-20, 25, 20)]
titles = ['SSI', 'Act', 'RFI', 'Tau']
yLabels = ['Normalised Peak', 'Normalised Steady State', 'Ratio of Peaks', 'Time to 50% Current (ms)']
plot_moreno(axsDict, xLabels, xCoords, titles, yLabels, xTicks)
fig.suptitle('Moreno INa')
fig.align_labels()
fig.savefig(
    os.path.join(ionbench.ROOT_DIR, '..', 'scripts', 'figures', f'data_moreno' + '.png'),
    bbox_inches='tight', dpi=300)
plt.show()

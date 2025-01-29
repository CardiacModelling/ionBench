import matplotlib.pyplot as plt
import numpy as np

problems = ("Staircase HH", "Staircase MM", "Loewe IKr", "Loewe IKur", "Moreno INa")
ert = {
    "New Approach": (502, 17405, 741, 97600, 5176),
    "Wilhelms2012b": (np.nan, 64839, 1376, 601502, 6584),
    "Dokos2004": (693, np.nan, np.nan, np.nan, np.nan)
}
colours = ['#DBB40C', '#1F77B4', '#B31E1E']

x = np.arange(len(problems))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained', figsize=(6, 4))

counter = 0
for attribute, measurement in ert.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute, color=colours[counter])
    counter += 1
    if multiplier == 0:
        multiplier = 1

ax.set_ylabel('ERT (FEs)')
ax.set_title('Evaluation of the new approach')
ax.set_xticks(x + width/2, problems)
ax.legend(loc='upper left', ncols=1)
ax.set_yscale('log')
ax.grid(axis='y')
ax.set_ylim(10, 1e6)

fig.savefig('figures/newAppFigure.png', dpi=300)
plt.show()

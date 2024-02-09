import ionbench
import numpy as np
import matplotlib.pyplot as plt


def solver_noise(bm, nPoints, intervalWidth):
    p = bm.defaultParams
    costs = []
    for i in range(nPoints):
        trialP = np.copy(p)
        trialP[0] *= 1+(i-nPoints/2)/nPoints*intervalWidth
        costs.append(bm.cost(trialP))
    plt.figure()
    plt.plot(costs)
    plt.title(f'Solver noise for {bm._name}')
    plt.show()

bm = ionbench.problems.staircase.HH()
solver_noise(bm, 100, 1e-6)

bm = ionbench.problems.staircase.MM()
solver_noise(bm, 100, 1e-7)

bm = ionbench.problems.loewe2016.ikr()
solver_noise(bm, 100, 1e-15)

bm = ionbench.problems.loewe2016.ikur()
solver_noise(bm, 100, 1e-15)

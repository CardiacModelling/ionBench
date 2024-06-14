import os
import ionbench


def test_multistart():
    # Check multistart runs (using lm scipy optimiser) and improves cost in all cases
    numStarts = 3
    bm = ionbench.problems.staircase.HH()
    bm.plotter = False
    initParams = bm.sample(n=numStarts)
    costs = []
    for x0 in initParams:
        costs.append(bm.cost(x0))
    outs = ionbench.utils.multistart(ionbench.optimisers.scipy_optimisers.nelderMead_scipy.run, bm, initParams, 'test', maxIter=5)
    for i in range(numStarts):
        assert bm.cost(outs[i]) <= costs[i]
        os.remove('test_run' + str(i) + '.pickle')


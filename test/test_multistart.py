import pytest
import ionbench

def test_multistart():
    #Check multistart runs (using lm scipy optimiser) and improves cost in all cases
    numStarts = 3
    bm = ionbench.problems.staircase.HH_Benchmarker()
    bm.plotter = False
    initParams = bm.sample(n=numStarts)
    costs = []
    for x0 in initParams:
        costs.append(bm.cost(x0))
    outs = ionbench.multistart(ionbench.optimisers.scipy_optimisers.lm_scipy.run, bm, initParams, '', maxfev=50)
    for i in range(numStarts):
        assert bm.cost(outs[i])<=costs[i]

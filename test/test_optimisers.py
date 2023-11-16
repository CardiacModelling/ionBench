import pytest
import ionbench
import numpy as np
from importlib import import_module
from inspect import signature


def sub_dict(d, keys):
    return dict((k, d[k]) for k in keys if k in d)


class Test_scipy:
    # Check all scipy optimisers run and improve cost compared to x0
    bm = ionbench.problems.staircase.HH_Benchmarker(sensitivities=True)
    bm._useScaleFactors = True  # Helps the stability of conjugate GD (avoiding big steps which lead to parameters that result in failed model solves)
    bm.plotter = False

    @pytest.mark.parametrize("opt", ionbench.OPT_SCIPY)
    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_cost_reduction(self, opt):
        x0 = self.bm.sample()
        cost = self.bm.cost(x0)
        # sample initial point, find its cost, optimise, find opt cost, assert reduction
        module = import_module(opt)
        x0_opt = module.run(self.bm, x0=x0, maxIter=5)
        cost_opt = self.bm.cost(x0_opt)
        self.bm.reset()
        assert cost_opt <= cost


class Test_pints:
    # Check all pints optimisers run and improve cost compared to x0
    bm = ionbench.problems.staircase.HH_Benchmarker()
    bm.plotter = False

    @pytest.mark.parametrize("opt", ionbench.OPT_PINTS)
    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_cost_reduction(self, opt):
        x0 = self.bm.sample()
        cost = self.bm.cost(x0)
        # sample initial point, find its cost, optimise, find opt cost, assert reduction
        module = import_module(opt)
        x0_opt = module.run(self.bm, x0=x0, maxIter=5)
        cost_opt = self.bm.cost(x0_opt)
        self.bm.reset()
        assert cost_opt <= cost


class Test_external:
    # Check all external optimisers run
    bm = ionbench.problems.staircase.HH_Benchmarker(sensitivities=True)
    bm._useScaleFactors = True
    bm.plotter = False

    @pytest.mark.parametrize("opt", ionbench.OPT_EXT)
    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_cost_reduction(self, opt):
        hyperparams = {'maxIter': 5, 'nGens': 5, 'maxfev': 50, 'n': 5, 'popSize': 20, 'maxInnerIter': 20}
        x0 = self.bm.sample()
        # sample initial point, find its cost, optimise, find opt cost, can't assert reduction - population algorithms sample around x0 not at x0
        module = import_module(opt)
        x0_opt = module.run(self.bm, x0=x0, **sub_dict(hyperparams, signature(module.run).parameters))
        cost_opt = self.bm.cost(x0_opt)
        self.bm.reset()
        assert cost_opt < np.inf

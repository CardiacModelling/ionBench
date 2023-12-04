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

    bmTest = ionbench.problems.test.Test()

    @pytest.mark.parametrize("opt", ionbench.OPT_SCIPY)
    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_with_problem(self, opt):
        x0 = self.bm.sample()
        # sample initial point, find its cost, optimise, find opt cost, assert not infinite
        module = import_module(opt)
        x0_opt = module.run(self.bm, x0=x0, maxIter=5)
        cost_opt = self.bm.cost(x0_opt)
        self.bm.reset()
        assert cost_opt < np.inf

    @pytest.mark.parametrize("opt", ionbench.OPT_SCIPY)
    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    @pytest.mark.cheap
    def test_minimum_finding(self, opt):
        module = import_module(opt)
        mod = module.get_modification()
        mod.apply(self.bmTest)
        x0_opt = module.run(self.bmTest)
        cost_opt = self.bmTest.cost(x0_opt)
        assert cost_opt < 5e-3


class Test_pints:
    # Check all pints optimisers run and improve cost compared to x0
    bm = ionbench.problems.staircase.HH_Benchmarker()
    bm.plotter = False

    bmTest = ionbench.problems.test.Test()

    @pytest.mark.parametrize("opt", ionbench.OPT_PINTS)
    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_with_problem(self, opt):
        x0 = self.bm.sample()
        # sample initial point, find its cost, optimise, find opt cost, assert not infinite
        module = import_module(opt)
        x0_opt = module.run(self.bm, x0=x0, maxIter=5)
        cost_opt = self.bm.cost(x0_opt)
        self.bm.reset()
        assert cost_opt < np.inf

    @pytest.mark.parametrize("opt", ionbench.OPT_PINTS)
    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    @pytest.mark.cheap
    def test_minimum_finding(self, opt):
        module = import_module(opt)
        mod = module.get_modification()
        mod.apply(self.bmTest)
        if 'cmaes' in opt:
            # Turn off bounds since rate bounds don't make sense here
            self.bmTest._bounded = False
        x0_opt = module.run(self.bmTest)
        cost_opt = self.bmTest.cost(x0_opt)
        assert cost_opt < 5e-3


class Test_external:
    # Check all external optimisers run
    bm = ionbench.problems.staircase.HH_Benchmarker(sensitivities=True)
    bm._useScaleFactors = True
    bm.plotter = False

    bmTest = ionbench.problems.test.Test()

    @pytest.mark.parametrize("opt", ionbench.OPT_EXT)
    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_with_problem(self, opt):
        hyperparams = {'maxIter': 5, 'nGens': 5, 'maxfev': 50, 'n': 5, 'popSize': 20, 'maxInnerIter': 20}
        x0 = self.bm.sample()
        # sample initial point, find its cost, optimise, find opt cost, assert not infinite
        module = import_module(opt)
        x0_opt = module.run(self.bm, x0=x0, **sub_dict(hyperparams, signature(module.run).parameters))
        cost_opt = self.bm.cost(x0_opt)
        self.bm.reset()
        assert cost_opt < np.inf

    @pytest.mark.parametrize("opt", ionbench.OPT_EXT)
    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    @pytest.mark.cheap
    def test_minimum_finding(self, opt):
        module = import_module(opt)
        mod = module.get_modification()
        mod.apply(self.bmTest)
        if 'Loewe' in opt:
            x0_opt = module.run(self.bmTest, maxIter=50)
        elif 'Liu' in opt or 'Chen' in opt or 'Cabo' in opt:
            x0_opt = module.run(self.bmTest, gmin=5e-3)
        elif 'Dokos' in opt:
            x0_opt = module.run(self.bmTest, costThreshold=1e-4)
        else:
            x0_opt = module.run(self.bmTest)
        cost_opt = self.bmTest.cost(x0_opt)
        assert cost_opt < 5e-3

import inspect

import pytest
import ionbench
import numpy as np
from importlib import import_module
from inspect import signature


def sub_dict(d, keys):
    return dict((k, d[k]) for k in keys if k in d)


class TestScipy:
    # Check all scipy optimisers run and improve cost compared to x0
    bmTest = ionbench.problems.test.Test()

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


class TestPints:
    bmTest = ionbench.problems.test.Test()

    @pytest.mark.parametrize("opt", ionbench.OPT_PINTS)
    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    @pytest.mark.cheap
    def test_minimum_finding(self, opt):
        module = import_module(opt)
        mod = module.get_modification()
        mod.apply(self.bmTest)
        if 'cmaes' in opt:
            # Turn off bounds since rate bounds don't make sense here
            self.bmTest._parameters_bounded = False
        x0_opt = module.run(self.bmTest)
        cost_opt = self.bmTest.cost(x0_opt)
        assert cost_opt < 5e-3


class TestExternal:
    bmTest = ionbench.problems.test.Test()

    @pytest.mark.parametrize("opt", ionbench.OPT_EXT)
    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    @pytest.mark.cheap
    def test_minimum_finding(self, opt):
        module = import_module(opt)
        mod = module.get_modification()
        mod.apply(self.bmTest)
        if 'maxIter' in inspect.signature(module.run).parameters:
            x0_opt = module.run(self.bmTest, maxIter=10000)
        else:
            x0_opt = module.run(self.bmTest)
        cost_opt = self.bmTest.cost(x0_opt)
        assert cost_opt < 5e-3

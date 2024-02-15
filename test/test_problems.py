import pytest
import ionbench
import numpy as np
import copy
import matplotlib.pyplot as plt
import myokit


class Problem:
    """
    General problem test class for specific problem test classes to inherit from.
    Defines tests for cost, attribute checks, plotting, parameter bounds, rate bounds, tracking, grad, and steady states.
    """

    @pytest.mark.cheap
    def test_cost(self):
        # Check cost of default params is sufficiently low (0 for loewe)
        assert self.bm.cost(self.bm.defaultParams) <= self.costBound

    @pytest.mark.cheap
    def test_hasattr(self):
        # Check all necessary variables in problems are defined
        assert hasattr(self.bm, "_name")
        assert hasattr(self.bm, "model")
        assert hasattr(self.bm, "_outputName")
        assert hasattr(self.bm, "_paramContainer")
        assert hasattr(self.bm, "defaultParams")
        assert hasattr(self.bm, "_rateFunctions")
        assert hasattr(self.bm, "standardLogTransform")
        assert hasattr(self.bm, "data")
        assert hasattr(self.bm, "sim")
        assert hasattr(self.bm, "_useScaleFactors")
        assert hasattr(self.bm, "_parameters_bounded")
        assert hasattr(self.bm, "_logTransformParams")
        assert hasattr(self.bm, "plotter")
        assert hasattr(self.bm, "tracker")
        assert hasattr(self.bm, "sensitivityCalc")
        assert hasattr(self.bm, "_analyticalModel")
        assert hasattr(self.bm, "tmax")
        assert hasattr(self.bm, "simSens")
        assert hasattr(self.bm, "freq")
        assert hasattr(self.bm, "rateMin")
        assert hasattr(self.bm, "rateMax")
        assert hasattr(self.bm, "vLow")
        assert hasattr(self.bm, "vHigh")
        assert hasattr(self.bm, "lbStandard")
        assert hasattr(self.bm, "ubStandard")

    @pytest.mark.cheap
    def test_plotter(self, monkeypatch):
        # Test plotting functionality runs for evaluate
        monkeypatch.setattr(plt, 'show', lambda: None)  # Disables plots from being shown
        self.bm.plotter = True
        self.bm.evaluate(self.bm.defaultParams)
        self.bm.plotter = False

    @pytest.mark.cheap
    def test_parameter_bounds(self):
        # Check bounds give infinite cost outside and solve inside bounds
        self.bm.add_parameter_bounds([0.99 * self.bm.defaultParams, [np.inf] * self.bm.n_parameters()])
        # Check bounds give penalty for 1 param out of bounds outside of bounds
        p = copy.copy(self.bm.defaultParams)
        p[0] *= 0.98
        c = self.bm.signed_error(p)[0] + self.bm.data
        assert 1e4 < c < 2e4
        # Check cost is small inside bounds when bounded
        p[0] = self.bm.defaultParams[0]
        assert self.bm.cost(p) < 1e4
        # Check _parameters_bounded = False turns off bounds
        self.bm._parameters_bounded = False
        p[0] *= 0.98
        assert self.bm.cost(p) < 1e4
        self.bm._parameter_bounded = True
        # Check penalty increases with more bound violations
        p = copy.copy(self.bm.defaultParams) * 0.98
        c = self.bm.signed_error(p)[0] + self.bm.data
        assert c > 1e4 * self.bm.n_parameters()
        # Check penalty increases linearly as parameters move out of bounds
        p = copy.copy(self.bm.defaultParams)
        p[2] *= 0.5  # Even if other parameters oob
        p[0] = 0.9 * self.bm.defaultParams[0]
        c1 = self.bm.signed_error(p)[0]
        p[0] = 0.8 * self.bm.defaultParams[0]
        c2 = self.bm.signed_error(p)[0]
        p[0] = 0.7 * self.bm.defaultParams[0]
        c3 = self.bm.signed_error(p)[0]
        assert np.abs(c2 - (c1 + c3) / 2) < 1e-12
        self.bm._parameter_bounded = False

    @pytest.mark.cheap
    def test_rate_bounds(self):
        assert self.bm._rates_bounded is False
        self.bm.add_rate_bounds()
        assert self.bm._rates_bounded is True
        tmp = self.bm.rateMin
        # Default params inside rate bounds always
        assert self.bm.cost(self.bm.defaultParams) < 1e4
        # Move rate bounds so default rates are outside
        self.bm.rateMin = self.bm.rateMax * 2
        assert self.bm.cost(self.bm.defaultParams) > 1e4
        # Turn off rate bounds should allow solving again (except for staircase)
        self.bm._rates_bounded = False
        assert self.bm.cost(self.bm.defaultParams) < 1e4 or 'staircase' in self.bm._name
        if 'staircase' in self.bm._name:
            assert self.bm.cost(self.bm.defaultParams) > 1e4
            self.rateMin = tmp
            assert self.bm.cost(self.bm.defaultParams) < 1e4
        self.bm.rateMin = tmp

    @pytest.mark.cheap
    def test_tracker(self):
        # Check solve count works with bounds and evaluate
        self.bm.reset()
        # Solve times are empty
        assert len(self.bm.tracker.costTimes) == 0
        assert len(self.bm.tracker.gradTimes) == 0
        # Solve count is 0 after reset
        assert self.bm.tracker.solveCount == 0
        # Solve count increments after solving
        self.bm.cost(self.bm.defaultParams)
        assert self.bm.tracker.solveCount == 1
        # but not when out of bounds
        self.bm.add_parameter_bounds([0.99 * self.bm.defaultParams, [np.inf] * self.bm.n_parameters()])
        p = copy.copy(self.bm.defaultParams)
        p[0] = 0.98 * self.bm.defaultParams[0]
        self.bm.cost(p)
        assert self.bm.tracker.solveCount == 1
        # or out of rate bounds
        self.bm._parameters_bounded = False
        self.bm.add_rate_bounds()
        tmp = self.bm.rateMin
        self.bm.rateMin = self.bm.rateMax
        self.bm.cost(self.bm.defaultParams)
        assert self.bm.tracker.solveCount == 1
        self.bm.rateMin = tmp
        self.bm.cost(self.bm.defaultParams)
        assert self.bm.tracker.solveCount == 2
        self.bm._rates_bounded = False
        # Solve count doesn't increment when evaluating
        self.bm.evaluate(p)
        assert self.bm.tracker.solveCount == 2
        # Only one cost time and no grad
        assert len(self.bm.tracker.costTimes) == 2
        assert len(self.bm.tracker.gradTimes) == 0
        # returns to 0 after resetting
        self.bm.reset(fullReset=False)
        assert self.bm.tracker.solveCount == 0
        self.bm._parameters_bounded = False
        if 'moreno' not in self.bm._name:
            # grad solve counter increments with grad but not normal solve counter
            self.bm.grad(p)
            assert self.bm.tracker.solveCount == 0
            assert self.bm.tracker.gradSolveCount == 1
            assert len(self.bm.tracker.costTimes) == 0
            assert len(self.bm.tracker.gradTimes) == 1
        self.bm.reset(fullReset=False)
        p1 = self.bm.sample()
        p2 = self.bm.sample()
        c1 = self.bm.cost(p1)
        c2 = self.bm.cost(p2)
        if c1 < c2:
            assert c1 == self.bm.tracker.bestCost
            assert all(p1 == self.bm.tracker.bestParams)
        else:
            assert c2 == self.bm.tracker.bestCost
            assert all(p2 == self.bm.tracker.bestParams)
        self.bm.cost(self.bm.defaultParams)
        assert all(self.bm.defaultParams == self.bm.tracker.bestParams)
        self.bm.reset()

    @pytest.mark.cheap
    def test_repeated_params_warning(self):
        self.bm.reset(fullReset=True)
        self.bm.cost(self.bm.defaultParams)
        self.bm.cost(self.bm.sample())
        with pytest.warns(UserWarning):
            self.bm.cost(self.bm.defaultParams)

    @pytest.mark.filterwarnings("ignore:Current:UserWarning")
    def test_grad(self, plotting=False):
        self.bm.use_sensitivities()
        # Check gradient calculator is accurate
        assert grad_check(bm=self.bm,
                          plotting=plotting) < 0.01  # Within 1% to account for solver noise and finite difference error
        # Same under log transforms
        self.bm.log_transform([True] + [False] * (self.bm.n_parameters() - 1))
        assert grad_check(bm=self.bm, plotting=plotting) < 0.01
        # Same under scale factor and log transforms
        self.bm._useScaleFactors = True
        assert grad_check(bm=self.bm, plotting=plotting) < 0.01
        # Same under scale factor only
        self.bm.log_transform([False] * self.bm.n_parameters())
        assert grad_check(bm=self.bm, plotting=plotting) < 0.01
        # Reset bm
        self.bm._useScaleFactors = False

    @pytest.mark.cheap
    def test_steady_state(self):
        # Test steady state is right for random parameters
        if 'moreno' not in self.bm._name:
            self.bm.reset()
            out = self.bm.simulate(parameters=self.bm.sample(), times=np.arange(0, self.bm.tmax, self.bm.freq))
            assert np.abs((out[0] - out[1])) < 1e-8
        self.bm.reset()
        p = self.bm.sample()
        self.bm.set_steady_state(p)
        initState = self.bm.sim.state()
        self.bm.set_params(p)
        self.bm.sim.run(1)
        newState = self.bm.sim.state()
        assert all([np.abs((newState[i] - initState[i])) < 1e-8 for i in range(len(initState))])
        # Check it also updates simSens
        if not self.bm.sensitivityCalc:
            self.bm.use_sensitivities()
        self.bm.reset()
        p = self.bm.sample()
        self.bm.set_steady_state(p)
        initState = self.bm.simSens.state()
        self.bm.set_params(p)
        self.bm.simSens.run(1)
        newState = self.bm.sim.state()
        assert all([np.abs((newState[i] - initState[i])) < 1e-8 for i in range(len(initState))])
        self.bm.reset(fullReset=True)
        # Check sensitivities are steady state as well
        p = self.bm.sample()
        self.bm.set_steady_state(p)
        self.bm.set_params(p)
        initSens = self.bm.simSens._s_state
        self.bm.simSens.run(1)
        newSens = self.bm.simSens._s_state
        # _s_state stays the same
        assert all([np.abs((newSens[i] - initSens[i])) < 1e-8 for i in range(len(initSens))])
        # _s_state is non-zero
        assert any(initSens != 0)
        # sens of current plots will also show the same
        _, e = self.bm.simSens.run(5, log_times=[0, 5])
        assert all(e[0, 0, :] - e[1, 0, :] < 1e-8)

    @pytest.mark.cheap
    def test_clamp(self):
        # Check parameters that start out of bounds get clamped to inside bounds
        p = copy.copy(self.bm.defaultParams)
        self.bm.add_parameter_bounds(0.99 * self.bm.defaultParams, 1.01 * self.bm.defaultParams)
        assert self.bm.in_parameter_bounds(p)
        p[0] = 0.5 * self.bm.defaultParams[0]
        assert not self.bm.in_parameter_bounds(p)
        assert self.bm.in_parameter_bounds(self.bm.clamp_parameters(p))
        # Check that they are at exactly the bounds and not in interior
        assert self.bm.clamp_parameters(p)[0] == self.bm.lb[0]
        # Check that clamping doesn't change the original parameter vector
        assert p[0] != self.bm.clamp_parameters(p)[0]

    @pytest.mark.cheap
    def test_n_parameters(self):
        n = self.bm.n_parameters()
        assert n > 0
        assert n == len(self.bm.defaultParams)
        assert n == len(self.bm.standardLogTransform)
        assert n == len(self.bm.lbStandard)
        assert n == len(self.bm.ubStandard)

    @pytest.mark.cheap
    def test_reset(self):
        self.bm.reset(fullReset=True)
        assert self.bm._parameters_bounded is False
        assert self.bm._rates_bounded is False
        assert all(self.bm._logTransformedParams == False)
        assert self._useScaleFactors is False
        assert self.bm.tracker.solveCount == 0
        mod = ionbench.modification.Clerx2019()
        mod.apply(self.bm)
        self.bm._useScaleFactor = True
        self.bm.cost(self.bm.input_parameter_space(self.bm.defaultParams))
        assert self.bm._parameters_bounded is True
        assert self.bm._rates_bounded is True
        assert any(self.bm._logTransformedParams == True)
        assert self._useScaleFactors is True
        assert self.bm.tracker.solveCount == 1
        self.bm.reset(fullReset=False)
        assert self.bm._parameters_bounded is True
        assert self.bm._rates_bounded is True
        assert any(self.bm._logTransformedParams == True)
        assert self._useScaleFactors is True
        assert self.bm.tracker.solveCount == 0
        self.bm.reset()  # Full reset is True by default
        assert self.bm._parameters_bounded is False
        assert self.bm._rates_bounded is False
        assert all(self.bm._logTransformedParams == False)
        assert self._useScaleFactors is False
        assert self.bm.tracker.solveCount == 0


class Staircase(Problem):
    @pytest.mark.cheap
    def test_sampler(self):
        # Check sampler is inside the right bounds and doesn't just return default rates, across all transforms
        # Sampler in bounds
        p = self.bm.sample()
        assert self.bm.in_parameter_bounds(p, boundedCheck=False) and self.bm.in_rate_bounds(p, boundedCheck=False)
        # Sampled different to default
        assert sampler_different(self.bm, self.bm.defaultParams)
        # Same for scale factor parameter space
        self.bm._useScaleFactors = True
        assert sampler_bounds(self.bm, self.bm.input_parameter_space(bm.lbStandard), self.bm.input_parameter_space(bm.ubStandard))
        assert sampler_different(self.bm, np.ones(self.bm.n_parameters()))
        self.bm._useScaleFactors = False
        # Same for log transformed space
        self.bm.log_transform()
        assert sampler_bounds(self.bm, self.bm.input_parameter_space(bm.lbStandard), self.bm.input_parameter_space(bm.ubStandard))
        assert sampler_different(self.bm, np.log(self.bm.defaultParams))
        self.bm.log_transform([False] * self.bm.n_parameters())
        # Same for scale factor and log transformed space
        self.bm.log_transform()
        self.bm._useScaleFactors = True
        assert sampler_bounds(self.bm, self.bm.input_parameter_space(bm.lbStandard), self.bm.input_parameter_space(bm.ubStandard))
        assert sampler_different(self.bm, np.zeros(self.bm.n_parameters()))
        self.bm.log_transform([False] * self.bm.n_parameters())
        self.bm._useScaleFactors = False

    @pytest.mark.cheap
    def test_transforms(self):
        # Check transforms map as expected
        # Log transform default rates
        self.bm.log_transform()
        assert param_equal(self.bm.input_parameter_space(self.bm.defaultParams), np.log(self.bm.defaultParams))
        # Original space
        assert param_equal(self.bm.original_parameter_space(np.log(self.bm.defaultParams)), self.bm.defaultParams)
        self.bm.log_transform([False] * self.bm.n_parameters())
        # Scale factor default rates
        self.bm._useScaleFactors = True
        assert param_equal(self.bm.input_parameter_space(self.bm.defaultParams), np.ones(self.bm.n_parameters()))
        assert param_equal(self.bm.original_parameter_space(np.ones(self.bm.n_parameters())), self.bm.defaultParams)
        # Scale factor and log transformed space
        self.bm.log_transform()
        assert param_equal(self.bm.input_parameter_space(self.bm.defaultParams), np.zeros(self.bm.n_parameters()))
        assert param_equal(self.bm.original_parameter_space(np.zeros(self.bm.n_parameters())), self.bm.defaultParams)
        self.bm._useScaleFactors = False
        self.bm.log_transform([False] * self.bm.n_parameters())


class Loewe(Problem):
    @pytest.mark.cheap
    def test_sampler(self):
        # Check sampler is inside the right bounds and doesn't just return default rates, across all transforms
        # Get bounds from a modification
        mod = ionbench.modification.Modification(parameterBounds='Sampler')
        mod.apply(self.bm)
        lb = self.bm.lb
        ub = self.bm.ub
        self.bm._parameters_bounded = False
        # Sampler in bounds
        assert sampler_bounds(self.bm, lb, ub)
        # Sampled different to default
        assert sampler_different(self.bm, self.bm.defaultParams)
        # Same for scale factor parameter space
        self.bm._useScaleFactors = True
        assert sampler_bounds(self.bm, self.bm.input_parameter_space(lb), self.bm.input_parameter_space(ub))
        assert sampler_different(self.bm, np.ones(self.bm.n_parameters()))
        self.bm._useScaleFactors = False
        # Same for log transformed space
        self.bm.log_transform([not i for i in self.bm.additiveParams])
        assert sampler_bounds(self.bm, self.bm.input_parameter_space(lb), self.bm.input_parameter_space(ub))
        assert sampler_different(self.bm, np.log(self.bm.defaultParams))
        self.bm.log_transform([False] * self.bm.n_parameters())
        # Same for scale factor and log transformed space
        self.bm.log_transform([not i for i in self.bm.additiveParams])
        self.bm._useScaleFactors = True
        assert sampler_bounds(self.bm, self.bm.input_parameter_space(lb), self.bm.input_parameter_space(ub))
        assert sampler_different(self.bm, np.zeros(self.bm.n_parameters()))
        self.bm.log_transform([False] * self.bm.n_parameters())
        self.bm._useScaleFactors = False

    @pytest.mark.cheap
    def test_transforms(self):
        # Check transforms map as expected
        # Log transform default rates
        self.bm.log_transform([not i for i in self.bm.additiveParams])
        logDefault = [np.log(self.bm.defaultParams[i]) if self.bm.standardLogTransform[i] else self.bm.defaultParams[i]
                      for i in range(self.bm.n_parameters())]
        assert param_equal(self.bm.input_parameter_space(self.bm.defaultParams), logDefault)
        # Original space
        assert param_equal(self.bm.original_parameter_space(logDefault), self.bm.defaultParams)
        self.bm.log_transform([False] * self.bm.n_parameters())
        # Scale factor default rates
        self.bm._useScaleFactors = True
        assert param_equal(self.bm.input_parameter_space(self.bm.defaultParams), np.ones(self.bm.n_parameters()))
        assert param_equal(self.bm.original_parameter_space(np.ones(self.bm.n_parameters())), self.bm.defaultParams)
        # Scale factor and log transformed space
        self.bm.log_transform([not i for i in self.bm.additiveParams])
        assert param_equal(self.bm.input_parameter_space(self.bm.defaultParams),
                           [0.0 if self.bm.standardLogTransform[i] else 1.0 for i in range(self.bm.n_parameters())])
        assert param_equal(self.bm.original_parameter_space(
            [0.0 if self.bm.standardLogTransform[i] else 1.0 for i in range(self.bm.n_parameters())]),
            self.bm.defaultParams)
        self.bm.log_transform([False] * self.bm.n_parameters())
        self.bm._useScaleFactors = False


class TestMoreno(Problem):
    bm = ionbench.problems.moreno2016.INa()
    bm.plotter = False
    costBound = 1e-4

    @pytest.mark.cheap
    def test_sampler(self):
        # Check sampler is inside the right bounds and doesn't just return default rates, across all transforms
        # Sampler in bounds
        assert sampler_bounds(self.bm, 0.75 * self.bm.defaultParams, 1.25 * self.bm.defaultParams)
        # Sampled different to default
        assert sampler_different(self.bm, self.bm.defaultParams)
        # Same for wider bounds
        self.bm.paramSpaceWidth = 90
        assert sampler_bounds(self.bm, 0.1 * self.bm.defaultParams, 1.9 * self.bm.defaultParams)
        self.bm.paramSpaceWidth = 25
        # Sampled different to default
        assert sampler_different(self.bm, self.bm.defaultParams)
        # Same for scale factor parameter space
        self.bm._useScaleFactors = True
        assert sampler_bounds(self.bm, 0.75, 1.25)
        assert sampler_different(self.bm, np.ones(self.bm.n_parameters()))
        self.bm._useScaleFactors = False
        # Same for log transformed space
        self.bm.log_transform()
        assert sampler_bounds(self.bm, np.log(0.75) + np.log(self.bm.defaultParams),
                              np.log(1.25) + np.log(self.bm.defaultParams))
        assert sampler_different(self.bm, np.log(self.bm.defaultParams))
        self.bm.log_transform([False] * self.bm.n_parameters())
        # Same for scale factor and log transformed space
        self.bm.log_transform()
        self.bm._useScaleFactors = True
        assert sampler_bounds(self.bm, np.log(0.75), np.log(1.25))
        assert sampler_different(self.bm, np.zeros(self.bm.n_parameters()))
        self.bm.log_transform([False] * self.bm.n_parameters())
        self.bm._useScaleFactors = False

    @pytest.mark.cheap
    def test_transforms(self):
        # Check transforms map as expected
        # Log transform default rates
        self.bm.log_transform()
        assert param_equal(self.bm.input_parameter_space(self.bm.defaultParams), np.log(self.bm.defaultParams))
        # Original space
        assert param_equal(self.bm.original_parameter_space(np.log(self.bm.defaultParams)), self.bm.defaultParams)
        self.bm.log_transform([False] * self.bm.n_parameters())
        # Scale factor default rates
        self.bm._useScaleFactors = True
        assert param_equal(self.bm.input_parameter_space(self.bm.defaultParams), np.ones(self.bm.n_parameters()))
        assert param_equal(self.bm.original_parameter_space(np.ones(self.bm.n_parameters())), self.bm.defaultParams)
        # Scale factor and log transformed space
        self.bm.log_transform()
        assert param_equal(self.bm.input_parameter_space(self.bm.defaultParams), np.zeros(self.bm.n_parameters()))
        assert param_equal(self.bm.original_parameter_space(np.zeros(self.bm.n_parameters())), self.bm.defaultParams)
        self.bm.log_transform([False] * self.bm.n_parameters())
        self.bm._useScaleFactors = False


class TestHH(Staircase):
    bm = ionbench.problems.staircase.HH()
    bm.plotter = False
    costBound = 0.04  # Accounts for noise


class TestMM(Staircase):
    bm = ionbench.problems.staircase.MM()
    bm.plotter = False
    costBound = 0.02  # Accounts for noise


class TestLoeweIKr(Loewe):
    bm = ionbench.problems.loewe2016.IKr()
    bm.plotter = False
    costBound = 0


class TestLoeweIKur(Loewe):
    bm = ionbench.problems.loewe2016.IKur()
    bm.plotter = False
    costBound = 0


def sampler_bounds(bm, lb, ub):
    """
    Test function for checking that sampled parameters lie within the expected bounds
    """
    p = bm.sample()
    return all(np.logical_and(p >= lb, p <= ub))


def sampler_different(bm, default):
    """
    Test function for checking that sampled parameters are different to the default parameters
    """
    p = bm.sample()
    return not param_equal(p, default)


def param_equal(p1, p2):
    """
    Test function to check if two parameters are equal
    """
    return all(np.abs(p1 - p2) < 1e-10)


def grad_check(bm, plotting=False):
    """
    Test function for checking gradient matches perturbing the cost function
    """
    x0 = bm.sample()
    if plotting:
        paramVec = np.linspace(0.999 * x0[0], 1.001 * x0[0], 10)
    else:
        paramVec = [0.999 * x0[0], 1.001 * x0[0]]
    nPoints = len(paramVec)
    costs = np.zeros(nPoints)
    for i in range(nPoints):
        p = np.copy(x0)
        p[0] = paramVec[i]
        costs[i] = bm.cost(p)
    grad = bm.grad(x0)

    centreCost = bm.cost(x0)
    if plotting:
        plt.plot(paramVec, costs)
        plt.plot(paramVec, centreCost + grad[0] * (paramVec - x0[0]))
        plt.legend(['ODE cost', 'Grad'])
        plt.show()
    actualGrad = (costs[-1] - costs[0]) / (paramVec[-1] - paramVec[0])
    return (actualGrad - grad[0]) ** 2 / actualGrad

import pytest
import ionbench
import numpy as np
import copy
import matplotlib.pyplot as plt
import myokit


class Problem():
    @pytest.mark.cheap
    def test_cost(self):
        # Check cost of default params is sufficiently low (0 for loewe)
        assert self.bm.cost(self.bm.defaultParams) <= self.costBound

    @pytest.mark.cheap
    def test_hasattr(self):
        # Check all neccessary variables in problems are defined
        assert hasattr(self.bm, "_name")
        assert hasattr(self.bm, "model")
        assert hasattr(self.bm, "_outputName")
        assert hasattr(self.bm, "_paramContainer")
        assert hasattr(self.bm, "defaultParams")
        assert hasattr(self.bm, "_rateFunctions")
        assert hasattr(self.bm, "standardLogTransform")
        assert hasattr(self.bm, "_trueParams")
        assert hasattr(self.bm, "data")
        assert hasattr(self.bm, "sim")
        assert hasattr(self.bm, "_useScaleFactors")
        assert hasattr(self.bm, "_bounded")
        assert hasattr(self.bm, "_logTransformParams")
        assert hasattr(self.bm, "plotter")
        assert hasattr(self.bm, "tracker")
        assert hasattr(self.bm, "sensitivityCalc")

    @pytest.mark.cheap
    def test_plotter(self, monkeypatch):
        # Test plotting functionality runs for evaluate
        monkeypatch.setattr(plt, 'show', lambda: None)  # Disables plots from being shown
        self.bm.plotter = True
        self.bm.evaluate(self.bm.defaultParams)
        self.bm.plotter = False

    @pytest.mark.cheap
    def test_bounds(self):
        # Check bounds give infinite cost outside and solve inside bounds
        self.bm.add_bounds([0.99 * self.bm.defaultParams, [np.inf] * self.bm.n_parameters()])
        # Check bounds give infinite cost outside of bounds
        p = copy.copy(self.bm.defaultParams)
        p[0] = 0.98 * p[0]
        assert self.bm.cost(p) == np.inf
        # Check cost is finite inside bounds when bounded
        p[0] = self.bm.defaultParams[0]
        assert self.bm.cost(p) < np.inf
        # Check _bounded = False turns off bounds
        self.bm._bounded = False
        p[0] = 0.98 * p[0]
        assert self.bm.cost(p) < np.inf

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
        self.bm.add_bounds([0.99 * self.bm.defaultParams, [np.inf] * self.bm.n_parameters()])
        p = copy.copy(self.bm.defaultParams)
        p[0] = 0.98 * self.bm.defaultParams[0]
        self.bm.cost(p)
        assert self.bm.tracker.solveCount == 1
        # Solve count doesn't increment when evaluating
        self.bm.evaluate(p)
        assert self.bm.tracker.solveCount == 1
        # Only one cost time and no grad
        assert len(self.bm.tracker.costTimes) == 1
        assert len(self.bm.tracker.gradTimes) == 0
        # returns to 0 after resetting
        self.bm.reset(fullReset = False)
        assert self.bm.tracker.solveCount == 0
        self.bm._bounded = False
        if 'moreno' not in self.bm._name:
            # grad solve counter increments with grad but not normal solve counter
            self.bm.grad(p)
            assert self.bm.tracker.solveCount == 0
            assert self.bm.tracker.gradSolveCount == 1
            assert len(self.bm.tracker.costTimes) == 0
            assert len(self.bm.tracker.gradTimes) == 1
        self.bm.reset(fullReset = False)
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

    @pytest.mark.filterwarnings("ignore:Current:UserWarning")
    def test_grad(self, plotting=False):
        self.bm.use_sensitivities()
        # Check gradient calculator is accurate
        assert grad_check(bm=self.bm, plotting=plotting) < 0.01  # Within 1% to account for solver noise
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
        self.bm.reset()


class Staircase(Problem):
    @pytest.mark.cheap
    def test_sampler(self):
        # Check sampler is inside the right bounds and doesnt just return default rates, across all transforms
        # Sampler in bounds
        assert sampler_bounds(self.bm, 0.5 * self.bm.defaultParams, 1.5 * self.bm.defaultParams)
        # Sampled different to default
        assert sampler_different(self.bm, self.bm.defaultParams)
        # Same for scale factor parameter space
        self.bm._useScaleFactors = True
        assert sampler_bounds(self.bm, 0.5, 1.5)
        assert sampler_different(self.bm, np.ones(self.bm.n_parameters()))
        self.bm._useScaleFactors = False
        # Same for log transformed space
        self.bm.log_transform()
        assert sampler_bounds(self.bm, np.log(0.5) + np.log(self.bm.defaultParams), np.log(1.5) + np.log(self.bm.defaultParams))
        assert sampler_different(self.bm, np.log(self.bm.defaultParams))
        self.bm.log_transform([False] * self.bm.n_parameters())
        # Same for scale factor and log transformed space
        self.bm.log_transform()
        self.bm._useScaleFactors = True
        assert sampler_bounds(self.bm, np.log(0.5), np.log(1.5))
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
        # Check sampler is inside the right bounds and doesnt just return default rates, across all transforms
        # Get bounds from a modification
        mod = ionbench.modification.Modification(bounds='Sampler')
        mod.apply(self.bm)
        lb = self.bm.lb
        ub = self.bm.ub
        self.bm._bounded = False
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
        logDefault = [np.log(self.bm.defaultParams[i]) if self.bm.standardLogTransform[i] else self.bm.defaultParams[i] for i in range(self.bm.n_parameters())]
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
        assert param_equal(self.bm.input_parameter_space(self.bm.defaultParams), [0.0 if self.bm.standardLogTransform[i] else 1.0 for i in range(self.bm.n_parameters())])
        assert param_equal(self.bm.original_parameter_space([0.0 if self.bm.standardLogTransform[i] else 1.0 for i in range(self.bm.n_parameters())]), self.bm.defaultParams)
        self.bm.log_transform([False] * self.bm.n_parameters())
        self.bm._useScaleFactors = False


class Test_Moreno(Problem):
    bm = ionbench.problems.moreno2016.ina()
    bm.plotter = False
    costBound = 1e-4

    @pytest.mark.cheap
    def test_sampler(self):
        # Check sampler is inside the right bounds and doesnt just return default rates, across all transforms
        # Sampler in bounds
        assert sampler_bounds(self.bm, 0.95 * self.bm.defaultParams, 1.05 * self.bm.defaultParams)
        # Sampled different to default
        assert sampler_different(self.bm, self.bm.defaultParams)
        # Same for wider bounds
        self.bm.paramSpaceWidth = 90
        assert sampler_bounds(self.bm, 0.1 * self.bm.defaultParams, 1.9 * self.bm.defaultParams)
        self.bm.paramSpaceWidth = 5
        # Sampled different to default
        assert sampler_different(self.bm, self.bm.defaultParams)
        # Same for scale factor parameter space
        self.bm._useScaleFactors = True
        assert sampler_bounds(self.bm, 0.95, 1.05)
        assert sampler_different(self.bm, np.ones(self.bm.n_parameters()))
        self.bm._useScaleFactors = False
        # Same for log transformed space
        self.bm.log_transform()
        assert sampler_bounds(self.bm, np.log(0.95) + np.log(self.bm.defaultParams), np.log(1.05) + np.log(self.bm.defaultParams))
        assert sampler_different(self.bm, np.log(self.bm.defaultParams))
        self.bm.log_transform([False] * self.bm.n_parameters())
        # Same for scale factor and log transformed space
        self.bm.log_transform()
        self.bm._useScaleFactors = True
        assert sampler_bounds(self.bm, np.log(0.95), np.log(1.05))
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

    @pytest.mark.cheap
    def test_steady_state(self):
        # Test steady state is right for random parameters
        self.bm.reset()
        p = self.bm.sample()
        self.bm.set_steady_state(p)
        self.bm.set_params(p)
        log = self.bm.sim.run(2, log_times=np.array([0, 1]))
        out = log[self.bm._outputName]
        assert np.abs((out[0] - out[1])) < 1e-8
        self.bm.reset()
        p = self.bm.sample()
        self.bm.set_steady_state(p)
        self.bm.set_params(p)
        initState = self.bm.sim.state()
        self.bm.set_params(p)
        self.bm.sim.run(1)
        newState = self.bm.sim.state()
        assert all([np.abs((newState[i] - initState[i])) < 1e-8 for i in range(len(initState))])


class Test_HH(Staircase):
    bm = ionbench.problems.staircase.HH_Benchmarker()
    bm.plotter = False
    costBound = 0.04  # Accounts for noise


class Test_MM(Staircase):
    bm = ionbench.problems.staircase.MM_Benchmarker()
    bm.plotter = False
    costBound = 0.02  # Accounts for noise


class Test_Loewe_IKr(Loewe):
    bm = ionbench.problems.loewe2016.ikr()
    bm.plotter = False
    costBound = 0


class Test_Loewe_IKur(Loewe):
    bm = ionbench.problems.loewe2016.ikur()
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
    return (actualGrad - grad[0])**2 / actualGrad

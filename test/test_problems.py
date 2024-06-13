import os
import pytest
import scipy.stats

import ionbench
import numpy as np
import matplotlib.pyplot as plt
import myokit


class Problem:
    """
    General problem test class for specific problem test classes to inherit from.
    Defines tests for cost, attribute checks, plotting, parameter bounds, rate bounds, tracking, grad, and steady states.
    """

    @pytest.mark.cheap
    def test_cost(self):
        self.bm.reset()
        # Check cost of default params is sufficiently low (0 for loewe)
        assert self.bm.cost(self.bm._TRUE_PARAMETERS) <= self.costBound
        # Check squared error works
        self.bm.squared_error(self.bm.sample())

    def test_evaluate(self):
        # No errors on empty evaluate
        self.bm.plotter = True
        self.bm.reset()
        self.bm.evaluate()
        self.bm.plotter = False

    @pytest.mark.cheap
    def test_hasattr(self):
        self.bm.reset()
        # Check all necessary variables in problems are defined
        assert hasattr(self.bm, "NAME")
        assert hasattr(self.bm, "_MODEL")
        assert hasattr(self.bm, "_OUTPUT_NAME")
        assert hasattr(self.bm, "_PARAMETER_CONTAINER")
        assert hasattr(self.bm, "_TRUE_PARAMETERS")
        assert hasattr(self.bm, "_RATE_FUNCTIONS")
        assert hasattr(self.bm, "STANDARD_LOG_TRANSFORM")
        assert hasattr(self.bm, "DATA")
        assert hasattr(self.bm, "sim")
        assert hasattr(self.bm, "useScaleFactors")
        assert hasattr(self.bm, "parametersBounded")
        assert hasattr(self.bm, "logTransformParams")
        assert hasattr(self.bm, "plotter")
        assert hasattr(self.bm, "tracker")
        assert hasattr(self.bm, "sensitivityCalc")
        assert hasattr(self.bm, "_ANALYTICAL_MODEL")
        assert hasattr(self.bm, "T_MAX")
        assert hasattr(self.bm, "simSens")
        assert hasattr(self.bm, "TIMESTEP")
        assert hasattr(self.bm, "RATE_MIN")
        assert hasattr(self.bm, "RATE_MAX")
        assert hasattr(self.bm, "V_LOW")
        assert hasattr(self.bm, "V_HIGH")
        assert hasattr(self.bm, "_LOWER_BOUND")
        assert hasattr(self.bm, "_UPPER_BOUND")

    @pytest.mark.cheap
    def test_plotter(self, monkeypatch):
        self.bm.reset()
        # Test plotting functionality runs for evaluate
        monkeypatch.setattr(plt, 'show', lambda: None)  # Disables plots from being shown
        self.bm.plotter = True
        self.bm.cost(self.bm.sample())
        self.bm.evaluate()
        self.bm.plotter = False

    @pytest.mark.cheap
    def test_parameter_bounds(self):
        self.bm.reset()
        # Check bounds give infinite cost outside and solve inside bounds
        self.bm.add_parameter_bounds()
        self.bm.lb = 0.99*self.bm._TRUE_PARAMETERS
        # Check bounds give penalty for 1 param out of bounds outside of bounds
        p = np.copy(self.bm._TRUE_PARAMETERS)
        p[0] *= 0.98
        c = self.bm.signed_error(p)[0]
        c2 = self.bm.cost(p)
        assert 1e5 < c < 2e5
        assert 1e5 < c2 < 2e5
        # Check cost is small inside bounds when bounded
        p[0] = self.bm._TRUE_PARAMETERS[0]
        assert self.bm.cost(p) < 1e5
        # Check parametersBounded = False turns off bounds
        self.bm.parametersBounded = False
        p[0] *= 0.98
        assert self.bm.cost(p) < 1e5
        self.bm.parametersBounded = True
        # Check penalty increases with more bound violations
        p = np.copy(self.bm._TRUE_PARAMETERS) * 0.98
        c = self.bm.signed_error(p)[0]
        assert c > 1e5 * self.bm.n_parameters()
        self.bm.parametersBounded = False

    @pytest.mark.cheap
    def test_rate_bounds(self):
        self.bm.reset()
        if 'staircase' in self.bm.NAME:
            assert self.bm.ratesBounded is True
        else:
            assert self.bm.ratesBounded is False
        self.bm.add_rate_bounds()
        assert self.bm.ratesBounded is True
        tmp = self.bm.RATE_MIN
        # Default params inside rate bounds always
        assert self.bm.cost(self.bm._TRUE_PARAMETERS) < 1e5
        # Move rate bounds so default rates are outside
        self.bm.RATE_MIN = self.bm.RATE_MAX * 2
        assert self.bm.cost(self.bm._TRUE_PARAMETERS) > 1e5
        # Turn off rate bounds should allow solving again
        self.bm.ratesBounded = False
        assert self.bm.cost(self.bm._TRUE_PARAMETERS) < 1e5
        self.bm.RATE_MIN = tmp
        # Test rate upper bounds
        self.bm.reset()
        self.bm.add_rate_bounds()
        p = self.bm.sample()
        self.bm.RATE_MAX = self.bm.RATE_MIN
        assert self.bm.cost(p) > 1e5

    @pytest.mark.cheap
    def test_tracker_solve_count(self):
        # Check solve count works with bounds and evaluate
        self.bm.reset()
        # Solve times are empty
        assert len(self.bm.tracker.costTimes) == 0
        assert len(self.bm.tracker.gradTimes) == 0
        # Solve count is 0 after reset
        assert self.bm.tracker.costSolveCount == 0
        # Solve count increments after solving
        self.bm.cost(self.bm._TRUE_PARAMETERS)
        assert self.bm.tracker.costSolveCount == 1
        # but not when out of bounds
        self.bm.add_parameter_bounds()
        self.bm.lb = 0.99*self.bm._TRUE_PARAMETERS
        p = np.copy(self.bm._TRUE_PARAMETERS)
        p[0] = 0.98 * self.bm._TRUE_PARAMETERS[0]
        self.bm.cost(p)
        assert self.bm.tracker.costSolveCount == 1
        self.bm.parametersBounded = False
        # or out of rate bounds
        self.bm.add_rate_bounds()
        tmp = self.bm.RATE_MIN
        self.bm.RATE_MIN = self.bm.RATE_MAX
        self.bm.cost(self.bm._TRUE_PARAMETERS)
        assert self.bm.tracker.costSolveCount == 1
        self.bm.RATE_MIN = tmp
        self.bm.cost(self.bm._TRUE_PARAMETERS*(1+1e-12))
        assert self.bm.tracker.costSolveCount == 2
        self.bm.ratesBounded = False
        # Solve count doesn't increment when evaluating
        self.bm.evaluate()
        assert self.bm.tracker.costSolveCount == 2
        assert self.bm.tracker.gradSolveCount == 0
        # returns to 0 after resetting
        self.bm.reset(fullReset=False)
        assert self.bm.tracker.costSolveCount == 0
        self.bm.parametersBounded = False
        # grad solve counter increments with grad but not normal solve counter
        self.bm.grad(self.bm._TRUE_PARAMETERS)
        assert self.bm.tracker.costSolveCount == 0
        assert self.bm.tracker.gradSolveCount == 1
        self.bm.reset(fullReset=False)
        p1 = self.bm.sample()
        p2 = self.bm.sample()
        c1 = self.bm.cost(p1)
        c2 = self.bm.cost(p2)
        if c1 < c2:
            assert np.abs(c1 - self.bm.tracker.bestCost) < 1e-8
            assert all(p1 == self.bm.tracker.bestParams)
        else:
            assert np.abs(c2 - self.bm.tracker.bestCost) < 1e-8
            assert all(p2 == self.bm.tracker.bestParams)
        self.bm.cost(self.bm._TRUE_PARAMETERS)
        assert all(self.bm._TRUE_PARAMETERS == self.bm.tracker.bestParams)
        self.bm.reset()

    def test_tracker_save_load(self):
        self.bm.reset()
        p = self.bm.sample()
        self.bm.cost(p)
        cost = np.copy(self.bm.tracker.costs[0])
        time = np.copy(self.bm.tracker.costTimes[0])
        self.bm.tracker.save('temporary_test_tracker.pickle')
        self.bm.reset()
        self.bm.tracker.load('temporary_test_tracker.pickle')
        assert cost == self.bm.tracker.costs[0]
        assert time == self.bm.tracker.costTimes[0]
        os.remove('temporary_test_tracker.pickle')

    @pytest.mark.cheap
    def test_convergence(self):
        self.bm.reset()
        p = self.bm.sample()
        c = self.bm.cost(p)
        if c < self.bm.COST_THRESHOLD:
            assert self.bm.is_converged()
            assert self.bm.tracker.cost_threshold(self.bm.COST_THRESHOLD)
        else:
            assert not self.bm.is_converged()
            assert not self.bm.tracker.cost_threshold(self.bm.COST_THRESHOLD)
            self.bm.cost(self.bm._TRUE_PARAMETERS * (1+1e-7))  # Test that non-default parameters give convergence
            assert self.bm.is_converged()
            assert self.bm.tracker.cost_threshold(self.bm.COST_THRESHOLD)
        # Dropping the cost threshold means not converged
        tmp = self.bm.COST_THRESHOLD
        self.bm.COST_THRESHOLD = -1
        assert not self.bm.is_converged()
        assert not self.bm.tracker.cost_threshold(self.bm.COST_THRESHOLD)
        self.bm.COST_THRESHOLD = tmp
        self.bm.reset()
        # Tracker initially not converged
        assert not self.bm.is_converged()
        assert not self.bm.tracker.cost_threshold(self.bm.COST_THRESHOLD)
        assert not self.bm.tracker.cost_unchanged()
        # Cost convergence rather than threshold
        self.bm.reset()
        # Get a point that doesn't satisfy the cost threshold
        p = self.bm.sample()
        while self.bm.cost(p) <= self.bm.COST_THRESHOLD:
            p = self.bm.sample()
        # Shouldn't be converged from this
        assert not self.bm.is_converged()
        assert not self.bm.tracker.cost_threshold(self.bm.COST_THRESHOLD)
        self.bm.reset()
        # Generate 6 random points and order them by cost
        points = [self.bm.sample() for _ in range(6)]
        costs = [self.bm.cost(p) for p in points]
        order = np.argsort(costs)
        points = [points[i] for i in order]
        self.bm.reset()
        # First point is an improvement
        self.bm.cost(points[0])
        # 2 points without improvement aren't enough
        self.bm.cost(points[1])
        self.bm.cost(points[2])
        assert not self.bm.tracker.cost_unchanged(max_unchanged_evals=3)
        # Third point is enough
        self.bm.cost(points[3])
        assert self.bm.tracker.cost_unchanged(max_unchanged_evals=3)
        # Sampling better points resets counter
        self.bm.reset()
        self.bm.cost(points[1])
        self.bm.cost(points[2])
        assert not self.bm.tracker.cost_unchanged(max_unchanged_evals=3)
        c1 = self.bm.cost(points[0])
        assert not self.bm.tracker.cost_unchanged(max_unchanged_evals=3)
        # Three worse points from here needed to converge
        [self.bm.cost(p) for p in points[3:5]]
        assert not self.bm.tracker.cost_unchanged(max_unchanged_evals=3)
        self.bm.cost(points[5])
        assert self.bm.tracker.cost_unchanged(max_unchanged_evals=3)
        # Sampling better points doesn't undo convergence
        self.bm.cost(self.bm._TRUE_PARAMETERS)
        assert self.bm.tracker.cost_unchanged(max_unchanged_evals=3)
        assert self.bm.is_converged()
        self.bm.reset()

    @pytest.mark.cheap
    def test_check_repeated_params(self):
        self.bm.reset()
        self.bm.reset(fullReset=True)
        self.bm.cost(self.bm._TRUE_PARAMETERS)
        self.bm.cost(self.bm.sample())
        assert len(self.bm.tracker.evals) == 2
        self.bm.cost(self.bm._TRUE_PARAMETERS)
        assert len(self.bm.tracker.evals) == 2
        assert self.bm.tracker.check_repeated_param(self.bm._TRUE_PARAMETERS, 'cost')

    @pytest.mark.filterwarnings("ignore:Current:UserWarning")
    def test_grad(self, plotting=False):
        self.bm.reset()
        self.bm.use_sensitivities()
        # Check gradient calculator is accurate
        a = grad_check(bm=self.bm, x0=self.bm.sample(), plotting=plotting)
        assert a < 0.01
        # Same under log transforms
        self.bm.log_transform([True] + [False] * (self.bm.n_parameters() - 1))
        assert grad_check(bm=self.bm, x0=self.bm.sample(), plotting=plotting) < 0.01
        # Same under scale factor and log transforms
        self.bm.useScaleFactors = True
        assert grad_check(bm=self.bm, x0=self.bm.sample(), plotting=plotting) < 0.01
        # Same under scale factor only
        self.bm.log_transform([False] * self.bm.n_parameters())
        assert grad_check(bm=self.bm, x0=self.bm.sample(), plotting=plotting) < 0.01
        # Input space = False
        self.bm.grad(self.bm.sample(), inInputSpace=False)

    @pytest.mark.filterwarnings("ignore:Current:UserWarning")
    def test_grad_bounds(self, plotting=False):
        self.bm.reset()
        mod = ionbench.modification.Modification(parameterBounds='on')
        mod.apply(self.bm)
        self.bm.use_sensitivities()
        x0 = self.bm.sample()
        # Put two parameter outside of bounds
        self.bm.lb[0] = self.bm.ub[0]
        self.bm.lb[2] = self.bm.ub[2]
        # Check gradient calculator is accurate when outside of bounds
        a = grad_check(bm=self.bm, x0=x0, plotting=plotting)
        assert a < 0.01
        # Handles rate bounds as well
        self.bm.add_rate_bounds()
        self.bm.lb = np.copy(self.bm._LOWER_BOUND)
        tmp = self.bm.RATE_MIN
        self.bm.RATE_MIN = self.bm.RATE_MAX
        a = grad_check(bm=self.bm, x0=x0, plotting=plotting)
        assert a < 0.01
        self.bm.useScaleFactors = True
        a = grad_check(bm=self.bm, x0=self.bm.input_parameter_space(x0), plotting=plotting)
        assert a < 0.01
        self.bm.RATE_MIN = tmp

    @pytest.mark.cheap
    def test_steady_state(self):
        self.bm.reset()
        # Test steady state is right for random parameters
        if 'moreno' not in self.bm.NAME:
            self.bm.reset()
            p = self.bm.sample()
            assert self.bm.in_parameter_bounds(p)
            assert self.bm.in_rate_bounds(p)
            out = self.bm.simulate(parameters=p, times=np.arange(0, self.bm.T_MAX, self.bm.TIMESTEP))
            assert np.abs((out[0] - out[1])) < 1e-8
        self.bm.reset()
        p = self.bm.sample()
        self.bm.set_steady_state(p)
        initState = self.bm.sim.state()
        self.bm.set_params(p)
        self.bm.sim.run(1)
        newState = self.bm.sim.state()
        assert all(np.abs(np.subtract(newState, initState)) < 1e-8)
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
        assert all(np.abs(np.subtract(newState, initState)) < 1e-8)
        self.bm.reset(fullReset=True)
        # Check sensitivities are steady state as well
        p = self.bm.sample()
        self.bm.set_steady_state(p)
        self.bm.set_params(p)
        initSens = self.bm.simSens._s_state
        if 'loewe' in self.bm.NAME:
            self.bm.simSens.set_tolerance(1e-7, 1e-7)
        self.bm.simSens.run(5)
        newSens = self.bm.simSens._s_state
        # _s_state stays the same
        assert np.all(np.abs(np.subtract(newSens, initSens)) <= 1e-5)
        # _s_state is non-zero
        assert not np.all(np.array(initSens) == 0)
        # sens of current plots will also show the same
        self.bm.reset()
        self.bm.set_params(p)
        self.bm.set_steady_state(p)
        _, e = self.bm.simSens.run(6, log_times=[0, 5])
        e = np.array(e)
        assert all(e[0, 0, :] - e[1, 0, :] < 1e-5)
        if 'loewe' in self.bm.NAME:
            self.bm.simSens.set_tolerance()

    @pytest.mark.cheap
    def test_clamp(self):
        self.bm.reset()
        # Check parameters that start out of bounds get clamped to inside bounds
        p = np.copy(self.bm._TRUE_PARAMETERS)
        self.bm.add_parameter_bounds()
        self.bm.lb = 0.99*self.bm._TRUE_PARAMETERS
        assert self.bm.in_parameter_bounds(p)
        p[0] = 0.5 * self.bm._TRUE_PARAMETERS[0]
        assert not self.bm.in_parameter_bounds(p)
        assert self.bm.in_parameter_bounds(self.bm.clamp_parameters(p))
        # Check that they are at exactly the bounds and not in interior
        assert self.bm.clamp_parameters(p)[0] == self.bm.lb[0]
        # Check that clamping doesn't change the original parameter vector
        assert p[0] != self.bm.clamp_parameters(p)[0]

    @pytest.mark.cheap
    def test_n_parameters(self):
        self.bm.reset()
        n = self.bm.n_parameters()
        assert n > 0
        assert n == len(self.bm._TRUE_PARAMETERS)
        assert n == len(self.bm.STANDARD_LOG_TRANSFORM)
        assert n == len(self.bm._LOWER_BOUND)
        assert n == len(self.bm._UPPER_BOUND)

    @pytest.mark.cheap
    def test_reset(self):
        self.bm.reset(fullReset=True)
        if 'staircase' in self.bm.NAME:
            assert self.bm.parametersBounded is True
            assert self.bm.ratesBounded is True
        else:
            assert self.bm.parametersBounded is False
            assert self.bm.ratesBounded is False
        assert all([self.bm.logTransformParams[i] is False for i in range(self.bm.n_parameters())])
        assert self.bm.useScaleFactors is False
        assert self.bm.tracker.costSolveCount == 0
        mod = ionbench.modification.Clerx2019()
        mod.apply(self.bm)
        self.bm.useScaleFactors = True
        if 'ikur' in self.bm.NAME:
            self.bm.ratesBounded = False
            self.bm.cost(self.bm.input_parameter_space(self.bm._TRUE_PARAMETERS))
            self.bm.ratesBounded = True
        else:
            self.bm.cost(self.bm.input_parameter_space(self.bm._TRUE_PARAMETERS))
        assert self.bm.parametersBounded is True
        assert self.bm.ratesBounded is True
        assert any([self.bm.logTransformParams[i] is True for i in range(self.bm.n_parameters())])
        assert self.bm.useScaleFactors is True
        assert self.bm.tracker.costSolveCount == 1
        self.bm.reset(fullReset=False)
        assert self.bm.parametersBounded is True
        assert self.bm.ratesBounded is True
        assert any([self.bm.logTransformParams[i] is True for i in range(self.bm.n_parameters())])
        assert self.bm.useScaleFactors is True
        assert self.bm.tracker.costSolveCount == 0
        self.bm.reset()  # Full reset is True by default
        if 'staircase' in self.bm.NAME:
            assert self.bm.parametersBounded is True
            assert self.bm.ratesBounded is True
        else:
            assert self.bm.parametersBounded is False
            assert self.bm.ratesBounded is False
        assert all([self.bm.logTransformParams[i] is False for i in range(self.bm.n_parameters())])
        assert self.bm.useScaleFactors is False
        assert self.bm.tracker.costSolveCount == 0
        assert self.bm.tracker.total_solve_time(i=None) == (0, 0)


class Staircase(Problem):
    @pytest.mark.cheap
    def test_sampler(self):
        self.bm.reset()
        # Check sampler is inside the right bounds and doesn't just return default rates, across all transforms
        # Sampler in bounds
        p = self.bm.sample()
        assert self.bm.in_parameter_bounds(p, boundedCheck=False) and self.bm.in_rate_bounds(p, boundedCheck=False)
        # Sampled different to default
        assert sampler_different(self.bm, self.bm._TRUE_PARAMETERS)
        # Same for scale factor parameter space
        self.bm.useScaleFactors = True
        assert sampler_bounds(self.bm, self.bm.input_parameter_space(self.bm._LOWER_BOUND),
                              self.bm.input_parameter_space(self.bm._UPPER_BOUND))
        assert sampler_different(self.bm, np.ones(self.bm.n_parameters()))
        self.bm.useScaleFactors = False
        # Same for log transformed space
        self.bm.log_transform()
        assert sampler_bounds(self.bm, self.bm.input_parameter_space(self.bm._LOWER_BOUND),
                              self.bm.input_parameter_space(self.bm._UPPER_BOUND))
        assert sampler_different(self.bm, np.log(self.bm._TRUE_PARAMETERS))
        self.bm.log_transform([False] * self.bm.n_parameters())
        # Same for scale factor and log transformed space
        self.bm.log_transform()
        self.bm.useScaleFactors = True
        assert sampler_bounds(self.bm, self.bm.input_parameter_space(self.bm._LOWER_BOUND),
                              self.bm.input_parameter_space(self.bm._UPPER_BOUND))
        assert sampler_different(self.bm, np.zeros(self.bm.n_parameters()))
        self.bm.log_transform([False] * self.bm.n_parameters())
        self.bm.useScaleFactors = False
        # n>1
        self.bm.reset()
        assert len(self.bm.sample(5)) == 5

    @pytest.mark.cheap
    def test_transforms(self):
        self.bm.reset()
        # Check transforms map as expected
        # Log transform default rates
        self.bm.log_transform()
        assert param_equal(self.bm.input_parameter_space(self.bm._TRUE_PARAMETERS), np.log(self.bm._TRUE_PARAMETERS))
        # Original space
        assert param_equal(self.bm.original_parameter_space(np.log(self.bm._TRUE_PARAMETERS)), self.bm._TRUE_PARAMETERS)
        self.bm.log_transform([False] * self.bm.n_parameters())
        # Scale factor default rates
        self.bm.useScaleFactors = True
        assert param_equal(self.bm.input_parameter_space(self.bm._TRUE_PARAMETERS), np.ones(self.bm.n_parameters()))
        assert param_equal(self.bm.original_parameter_space(np.ones(self.bm.n_parameters())), self.bm._TRUE_PARAMETERS)
        # Scale factor and log transformed space
        self.bm.log_transform()
        assert param_equal(self.bm.input_parameter_space(self.bm._TRUE_PARAMETERS), np.zeros(self.bm.n_parameters()))
        assert param_equal(self.bm.original_parameter_space(np.zeros(self.bm.n_parameters())), self.bm._TRUE_PARAMETERS)
        self.bm.useScaleFactors = False
        self.bm.log_transform([False] * self.bm.n_parameters())


class Loewe(Problem):
    @pytest.mark.cheap
    def test_sampler(self):
        self.bm.reset()
        # Check sampler is inside the right bounds and doesn't just return default rates, across all transforms
        # Get bounds from a modification
        mod = ionbench.modification.Modification(parameterBounds='on')
        mod.apply(self.bm)
        lb = self.bm.lb
        ub = self.bm.ub
        self.bm.parametersBounded = False
        # Sampler in bounds
        assert sampler_bounds(self.bm, lb, ub)
        # Sampled different to default
        assert sampler_different(self.bm, self.bm._TRUE_PARAMETERS)
        # Same for scale factor parameter space
        self.bm.useScaleFactors = True
        assert sampler_bounds(self.bm, self.bm.input_parameter_space(lb), self.bm.input_parameter_space(ub))
        assert sampler_different(self.bm, np.ones(self.bm.n_parameters()))
        self.bm.useScaleFactors = False
        # Same for log transformed space
        self.bm.log_transform(self.bm.STANDARD_LOG_TRANSFORM)
        assert sampler_bounds(self.bm, self.bm.input_parameter_space(lb), self.bm.input_parameter_space(ub))
        assert sampler_different(self.bm, np.log(self.bm._TRUE_PARAMETERS))
        self.bm.log_transform([False] * self.bm.n_parameters())
        # Same for scale factor and log transformed space
        self.bm.log_transform(self.bm.STANDARD_LOG_TRANSFORM)
        self.bm.useScaleFactors = True
        assert sampler_bounds(self.bm, self.bm.input_parameter_space(lb), self.bm.input_parameter_space(ub))
        assert sampler_different(self.bm, np.zeros(self.bm.n_parameters()))
        self.bm.log_transform([False] * self.bm.n_parameters())
        self.bm.useScaleFactors = False
        # n>1
        self.bm.reset()
        assert len(self.bm.sample(5)) == 5

    @pytest.mark.cheap
    def test_transforms(self):
        self.bm.reset()
        # Check transforms map as expected
        # Log transform default rates
        self.bm.log_transform(self.bm.STANDARD_LOG_TRANSFORM)
        logDefault = [np.log(self.bm._TRUE_PARAMETERS[i]) if self.bm.STANDARD_LOG_TRANSFORM[i] else self.bm._TRUE_PARAMETERS[i]
                      for i in range(self.bm.n_parameters())]
        assert param_equal(self.bm.input_parameter_space(self.bm._TRUE_PARAMETERS), logDefault)
        # Original space
        assert param_equal(self.bm.original_parameter_space(logDefault), self.bm._TRUE_PARAMETERS)
        self.bm.log_transform([False] * self.bm.n_parameters())
        # Scale factor default rates
        self.bm.useScaleFactors = True
        assert param_equal(self.bm.input_parameter_space(self.bm._TRUE_PARAMETERS), np.ones(self.bm.n_parameters()))
        assert param_equal(self.bm.original_parameter_space(np.ones(self.bm.n_parameters())), self.bm._TRUE_PARAMETERS)
        # Scale factor and log transformed space
        self.bm.log_transform(self.bm.STANDARD_LOG_TRANSFORM)
        assert param_equal(self.bm.input_parameter_space(self.bm._TRUE_PARAMETERS),
                           [0.0 if self.bm.STANDARD_LOG_TRANSFORM[i] else 1.0 for i in range(self.bm.n_parameters())])
        assert param_equal(self.bm.original_parameter_space(
            [0.0 if self.bm.STANDARD_LOG_TRANSFORM[i] else 1.0 for i in range(self.bm.n_parameters())]),
            self.bm._TRUE_PARAMETERS)
        self.bm.log_transform([False] * self.bm.n_parameters())
        self.bm.useScaleFactors = False


class TestMoreno(Problem):
    bm = ionbench.problems.moreno2016.INa(sensitivities=True)
    bm.plotter = False
    costBound = 1e-4

    @pytest.mark.cheap
    def test_sampler(self):
        self.bm.reset()
        # Check sampler is inside the right bounds and doesn't just return default rates, across all transforms
        # Sampler in bounds
        assert sampler_bounds(self.bm, 0.75 * self.bm._TRUE_PARAMETERS, 1.25 * self.bm._TRUE_PARAMETERS)
        # Sampled different to default
        assert sampler_different(self.bm, self.bm._TRUE_PARAMETERS)
        # Sampled different to default
        assert sampler_different(self.bm, self.bm._TRUE_PARAMETERS)
        # Same for scale factor parameter space
        self.bm.useScaleFactors = True
        assert sampler_bounds(self.bm, 0.75, 1.25)
        assert sampler_different(self.bm, np.ones(self.bm.n_parameters()))
        self.bm.useScaleFactors = False
        # Same for log transformed space
        self.bm.log_transform()
        assert sampler_bounds(self.bm, np.log(0.75) + np.log(self.bm._TRUE_PARAMETERS),
                              np.log(1.25) + np.log(self.bm._TRUE_PARAMETERS))
        assert sampler_different(self.bm, np.log(self.bm._TRUE_PARAMETERS))
        self.bm.log_transform([False] * self.bm.n_parameters())
        # Same for scale factor and log transformed space
        self.bm.log_transform()
        self.bm.useScaleFactors = True
        assert sampler_bounds(self.bm, np.log(0.75), np.log(1.25))
        assert sampler_different(self.bm, np.zeros(self.bm.n_parameters()))
        self.bm.log_transform([False] * self.bm.n_parameters())
        self.bm.useScaleFactors = False
        # n>1
        self.bm.reset()
        assert len(self.bm.sample(5)) == 5

    @pytest.mark.cheap
    def test_transforms(self):
        self.bm.reset()
        # Check transforms map as expected
        # Log transform default rates
        self.bm.log_transform()
        assert param_equal(self.bm.input_parameter_space(self.bm._TRUE_PARAMETERS), np.log(self.bm._TRUE_PARAMETERS))
        # Original space
        assert param_equal(self.bm.original_parameter_space(np.log(self.bm._TRUE_PARAMETERS)), self.bm._TRUE_PARAMETERS)
        self.bm.log_transform([False] * self.bm.n_parameters())
        # Scale factor default rates
        self.bm.useScaleFactors = True
        assert param_equal(self.bm.input_parameter_space(self.bm._TRUE_PARAMETERS), np.ones(self.bm.n_parameters()))
        assert param_equal(self.bm.original_parameter_space(np.ones(self.bm.n_parameters())), self.bm._TRUE_PARAMETERS)
        # Scale factor and log transformed space
        self.bm.log_transform()
        assert param_equal(self.bm.input_parameter_space(self.bm._TRUE_PARAMETERS), np.zeros(self.bm.n_parameters()))
        assert param_equal(self.bm.original_parameter_space(np.zeros(self.bm.n_parameters())), self.bm._TRUE_PARAMETERS)
        self.bm.log_transform([False] * self.bm.n_parameters())
        self.bm.useScaleFactors = False


class TestHH(Staircase):
    bm = ionbench.problems.staircase.HH(sensitivities=True)
    bm.plotter = False
    costBound = 0.04  # Accounts for noise


class TestMM(Staircase):
    bm = ionbench.problems.staircase.MM(sensitivities=True)
    bm.plotter = False
    costBound = 0.02  # Accounts for noise


class TestLoeweIKr(Loewe):
    bm = ionbench.problems.loewe2016.IKr(sensitivities=True)
    bm.plotter = False
    costBound = 1e-16


class TestLoeweIKur(Loewe):
    bm = ionbench.problems.loewe2016.IKur(sensitivities=True)
    bm.plotter = False
    costBound = 1e-16


class TestTest(Problem):
    bm = ionbench.problems.test.Test()
    bm.plotter = False
    costBound = 1e-16

    def test_grad_bounds(self, plotting=False):
        pass

    def test_steady_state(self):
        pass

    def test_rate_bounds(self):
        pass


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


def grad_check(bm, x0, plotting=False):
    """
    Test function for checking gradient matches perturbing the cost function
    """
    nPoints = 50
    paramVec = np.linspace(0.999 * x0[0], 1.001 * x0[0], nPoints)
    if 'staircase' in bm.NAME:
        bm.sim.set_tolerance(bm._TOLERANCES[0]/100, bm._TOLERANCES[1]/100)  # Use tighter tolerances here to avoid solver noise
    costs = np.zeros(nPoints)
    for i in range(nPoints):
        p = np.copy(x0)
        p[0] = paramVec[i]
        costs[i] = bm.cost(p)
    if 'staircase' in bm.NAME:
        bm.sim.set_tolerance(*bm._TOLERANCES)
    grad = bm.grad(x0)

    centreCost = bm.cost(x0)
    if plotting:
        plt.plot(paramVec, costs)
        plt.plot(paramVec, centreCost + grad[0] * (paramVec - x0[0]))
        plt.legend(['ODE cost', 'Grad'])
        plt.show()
    actualGrad, _, _, _, _ = scipy.stats.linregress(paramVec, costs)
    return (actualGrad - grad[0])**2 / actualGrad

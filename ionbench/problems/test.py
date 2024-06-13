"""
Contains the test benchmarker problem. This inherits from benchmarker.Benchmarker but with many of the methods overwritten. It is used to test the optimisers are able to sufficiently optimise on a problem that is much quicker to solve.
generate_data() will generate the data for test problem and store it in the data directory.
"""
import os
import numpy as np
import csv
import ionbench
from scipy.stats import norm
import warnings
import time

import ionbench.tracker.tracker


class Test(ionbench.benchmarker.Benchmarker):
    def __init__(self):
        self.NAME = "test"
        self.COST_THRESHOLD = 0.001
        self._TRUE_PARAMETERS = np.array([2., 4.])
        self._RATE_FUNCTIONS = ()
        self.RATE_MIN = 0
        self.RATE_MAX = 1
        self.STANDARD_LOG_TRANSFORM = (False, True)
        self.sensitivityCalc = True
        self.T_MAX = 20
        self.TIMESTEP = 1  # Timestep in data between points
        try:
            self.load_data(os.path.join(ionbench.DATA_DIR, 'test', 'data.csv'))
        except FileNotFoundError:  # pragma: no cover
            self.DATA = None
        self._LOWER_BOUND = self._TRUE_PARAMETERS * 0.5
        self._UPPER_BOUND = self._TRUE_PARAMETERS * 1.5
        self.lb = np.copy(self._LOWER_BOUND)
        self.ub = np.copy(self._UPPER_BOUND)
        self.sim = FakeSim()
        self.simSens = FakeSim()
        super().__init__()

    def sample(self, n=1):
        """
        Sample parameters for the test problem.

        Parameters
        ----------
        n : int, optional
            Number of parameter vectors to sample. The default is 1.

        Returns
        -------
        params : list
            If n=1, then params is the vector of parameters. Otherwise, params is a list containing n parameter vectors.

        """
        params = [None] * n
        for i in range(n):
            params[i] = self.input_parameter_space(
                self._TRUE_PARAMETERS * np.random.uniform(0.5, 1.5, self.n_parameters()))
        if n == 1:
            return params[0]
        else:
            return params

    def solve_model(self, times, continueOnError=True):
        out = norm(self.sim.parameters['.p1'], self.sim.parameters['.p2']).pdf(times)
        return out

    def solve_with_sensitivities(self, times):
        parameters = np.array([self.simSens.parameters['.p1'], self.simSens.parameters['.p2']])
        curr = norm(self.sim.parameters['.p1'], self.sim.parameters['.p2']).pdf(times)
        sens = np.zeros((len(times), 1, self.n_parameters()))
        for t in range(len(times)):
            sens[t, 0, 0] = curr[t] * (t - parameters[0]) / parameters[1] ** 2
            sens[t, 0, 1] = curr[t] * ((t - parameters[0]) ** 2 / parameters[1] ** 3 - 1 / parameters[1])
        return curr, sens

    def use_sensitivities(self):  # pragma: no cover
        # Not needed for test function. Override to avoid trying to access myokit objects
        pass

    def evaluate(self):  # pragma: no cover
        # Not needed for test function. Override to avoid trying to access myokit objects
        pass

    def set_steady_state(self, parameters):
        pass


class FakeSim:
    """
    A fake myokit simulation object to ignore any Simulation methods for the test problem.
    It stores set parameters.
    """
    def __init__(self):
        self.parameters = {}

    def reset(self):
        pass

    def set_constant(self, name, value):
        self.parameters[name] = value
        pass


# noinspection PyProtectedMember
def generate_data():  # pragma: no cover
    """
    Generate the data files for the test benchmarker problem.

    Parameters
    ----------
    None.

    Returns
    -------
    None.

    """
    bm = Test()
    out = bm.simulate(bm._TRUE_PARAMETERS, np.arange(0, bm.T_MAX, bm.TIMESTEP), continueOnError=False)
    with open(os.path.join(ionbench.DATA_DIR, 'test', 'data.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows(map(lambda x: [x], out))

"""
Contains the test benchmarker problem. This inherits from benchmarker.Benchmarker but with many of the methods overwritten. It is used to test the optimisers are able to sufficiently optimise on a problem that is much quicker to solve.
generate_data() will generate the data for test problem and store it in the data directory.
"""
import os
import numpy as np
import csv
import ionbench
import scipy.stats
import warnings

import ionbench.tracker.tracker


class Test(ionbench.benchmarker.Benchmarker):
    def __init__(self):
        self.NAME = "test"
        self.COST_THRESHOLD = 0.001
        self._TRUE_PARAMETERS = np.array([2, 4])
        self._RATE_FUNCTIONS = ()
        self.RATE_MIN = None
        self.RATE_MAX = None
        self.STANDARD_LOG_TRANSFORM = (False, True)
        self.sensitivityCalc = True
        self.T_MAX = 20
        self.freq = 1  # Timestep in data between points
        try:
            self.load_data(os.path.join(ionbench.DATA_DIR, 'test', 'data.csv'))
        except FileNotFoundError:
            self.DATA = None
        self._LOWER_BOUND = self._TRUE_PARAMETERS * 0.5
        self._UPPER_BOUND = self._TRUE_PARAMETERS * 1.5
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

    def simulate(self, parameters, times, continueOnError=True, incrementSolveCounter=True):
        # Calculate function at times for parameters and return
        parameters = self.original_parameter_space(parameters)
        if not self.in_parameter_bounds(parameters):
            return [np.inf for _ in times]
        return scipy.stats.norm(parameters[0], parameters[1]).pdf(times)

    def grad(self, parameters, incrementSolveCounter=True, inInputSpace=True, returnCost=False, residuals=False):
        # Calculate gradient wrt parameters
        # Undo any transforms
        curr = None
        sens = None
        J = None
        grad = None

        if inInputSpace:
            parameters = self.original_parameter_space(parameters)
        else:
            parameters = np.copy(parameters)

        # Abort solving if the parameters are out of bounds
        if not self.in_parameter_bounds(parameters):
            warnings.warn(
                'Tried to evaluate gradient when out of bounds. ionBench will try to resolve this by assuming infinite cost and a gradient that points back towards good parameters.')
            error = np.array([np.inf] * len(np.arange(0, self.T_MAX, self.freq)))
            # use grad to point back to reasonable parameter space
            grad = -1 / (self.original_parameter_space(self.sample()) - parameters)
            if residuals:
                J = np.zeros((len(error), self.n_parameters()))
                for i in range(len(error)):
                    J[i,] = grad
        else:
            # Get sensitivities
            curr = self.simulate(parameters, np.arange(0, self.T_MAX, self.freq))
            sens = np.zeros((len(curr), self.n_parameters()))
            for t in range(len(curr)):
                sens[t, 0] = curr[t] * (t - parameters[0]) / parameters[1] ** 2
                sens[t, 1] = curr[t] * ((t - parameters[0]) ** 2 / parameters[1] ** 3 - 1 / parameters[1])

        # Convert to cost derivative or residual jacobian
        error = curr - self.DATA
        cost = np.sqrt(np.mean(error ** 2))

        if residuals:
            J = sens
        else:
            grad = np.zeros(self.n_parameters())
            for i in range(self.n_parameters()):
                if cost > 0:
                    grad[i] = np.dot(error, sens[:, i]) / (len(error) * cost)
                else:
                    grad[i] = 0

        # Map derivatives to input space
        if inInputSpace:
            derivs = self.transform_jacobian(self.input_parameter_space(parameters))
            if residuals:
                J *= derivs
            else:
                grad *= derivs

        if returnCost:
            if residuals:
                return error, J
            else:
                return cost, grad
        else:
            if residuals:
                return J
            else:
                return grad

    def reset(self, fullReset=True):
        self.tracker = ionbench.tracker.Tracker(self._TRUE_PARAMETERS)
        if fullReset:
            self.log_transform([False] * self.n_parameters())
            self.useScaleFactors = False
            self.parametersBounded = False
            self.ratesBounded = False
            self.lb = np.copy(self._LOWER_BOUND)
            self.ub = np.copy(self._UPPER_BOUND)
            self.RATE_MIN = None
            self.RATE_MAX = None

    def use_sensitivities(self):
        # Not needed for test function. Override to avoid trying to access myokit objects
        pass

    def evaluate(self):
        # Not needed for test function. Override to avoid trying to access myokit objects
        pass


def generate_data():
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
    out = bm.simulate(bm._TRUE_PARAMETERS, np.arange(0, bm.T_MAX, bm.freq), continueOnError=False)
    with open(os.path.join(ionbench.DATA_DIR, 'test', 'data.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows(map(lambda x: [x], out))

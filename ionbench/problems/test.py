import os
import numpy as np
import csv
import ionbench
import scipy.stats
import warnings


class Test(ionbench.benchmarker.Benchmarker):
    def __init__(self):
        self._name = "test"
        self.defaultParams = np.array([2, 4])
        self._rateFunctions = [(lambda p, V:p[0] * np.exp(p[1] * V), 'positive'), (lambda p, V:p[2], 'independent'), (lambda p, V:p[3] * np.exp(p[4] * V), 'positive'), (lambda p, V:p[5] * np.exp(p[6] * V), 'positive'), (lambda p, V:p[7] * np.exp(-p[8] * V), 'negative'), (lambda p, V:p[9], 'independent'), (lambda p, V:p[10] * np.exp(-p[11] * V), 'negative'), (lambda p, V:p[12] * np.exp(-p[13] * V), 'negative')]  # Used for rate bounds
        self.standardLogTransform = [False, True]
        self.sensitivityCalc = True
        self._trueParams = np.copy(self.defaultParams)
        self.tmax = 20
        try:
            self.load_data(os.path.join(ionbench.DATA_DIR, 'test', 'data.csv'))
        except FileNotFoundError:
            self.data = None
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
            params[i] = self.input_parameter_space(self.defaultParams * np.random.uniform(0.5, 1.5, self.n_parameters()))
        if n == 1:
            return params[0]
        else:
            return params

    def simulate(self, parameters, times, continueOnError=True, incrementSolveCounter=True):
        # Calculate function at times for parameters and return
        parameters = self.original_parameter_space(parameters)
        if not self.in_bounds(parameters):
            return [np.inf for t in times]
        return scipy.stats.norm(parameters[0], parameters[1]).pdf(times)

    def grad(self, parameters, incrementSolveCounter=True, inInputSpace=True, returnCost=False, residuals=False):
        # Calculate gradient wrt parameters
        # Undo any transforms
        if inInputSpace:
            parameters = self.original_parameter_space(parameters)
        else:
            parameters = np.copy(parameters)

        # Abort solving if the parameters are out of bounds
        if not self.in_bounds(parameters):
            warnings.warn('Tried to evaluate gradient when out of bounds. ionBench will try to resolve this by assuming infinite cost and a gradient that points back towards good parameters.')
            error = np.array([np.inf] * len(np.arange(0, self.tmax)))
            cost = np.inf
            # use grad to point back to reasonable parameter space
            grad = -1 / (self.original_parameter_space(self.sample()) - parameters)
            if residuals:
                J = np.zeros((len(error), self.n_parameters()))
                for i in range(len(error)):
                    J[i, ] = grad
        else:
            # Get sensitivities
            sens = np.zeros((self.tmax, self.n_parameters()))
            curr = self.simulate(parameters, np.arange(self.tmax))
            for t in range(self.tmax):
                sens[t, 0] = curr[t] * (t - parameters[0]) / parameters[1]**2
                sens[t, 1] = curr[t] * ((t - parameters[0])**2 / parameters[1]**3 - 1 / parameters[1])

        # Convert to cost derivative or residual jacobian
        error = curr - self.data
        cost = np.sqrt(np.mean(error**2))

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
                J = J * derivs
            else:
                grad *= derivs

        if returnCost:
            if residuals:
                return (error, J)
            else:
                return (cost, grad)
        else:
            if residuals:
                return J
            else:
                return grad

    def reset(self):
        self.tracker = ionbench.benchmarker.Tracker(self._trueParams)

    def use_sensitivities(self):
        # Not needed for test function. Override to avoid trying to access myokit objects
        pass

    def evaluate(self, parameters):
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
    out = bm.simulate(bm._trueParams, np.arange(bm.tmax), continueOnError=False)
    with open(os.path.join(ionbench.DATA_DIR, 'test', 'data.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows(map(lambda x: [x], out))


if __name__ == '__main__':
    bm = Test()
    bm.grad(bm.sample())

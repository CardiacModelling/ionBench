"""
Contains the main Benchmarker class from which the benchmarker problems inherit the vast majority of their methods.
Contains the Tracker class which is stored as an attribute in each benchmarker and managers tracking the performance metrics over an optimisation.
"""
import numpy as np
import myokit
import csv
import ionbench
import matplotlib.pyplot as plt
import warnings
import pickle
import time
import mygrad as mg


class Tracker:
    """
    This class records the various performance metrics used to evaluate the optimisation algorithms.

    It records the number of times the model is solved (stored as tracker.solveCount), not including the times that parameters were out of bounds.

    It records the RMSE (Root Mean Squared Error) cost each time a parameter vector is evaluated, using np.inf if parameters are out of bounds.

    It records the RMSRE (Root Mean Squared Relative Error) of the estimated parameters each time a parameter vector is evaluated. This is relative to the true parameters, ie the RMS of the vector of error between the estimated and true parameters, expressed as a percentage of the true parameters.

    It records the number of parameters that were correctly identified (within 5% of the true values) each time a parameter vector is evaluated.

    This class contains two methods: update(), and plot().

    update() is called everytime a parameter vector is simulated in the benchmarker (for example, in bm.cost()) and updates the performance metric vectors.

    plot() is called during the benchmarker's .evaluate() method. It plots the performance metrics as functions of time (in the order in which parameter vectors were evaluated).
    """

    def __init__(self, trueParams):
        self.costs = []
        self.paramRMSRE = []
        self.paramIdentifiedCount = []
        self.solveCount = 0
        self.gradSolveCount = 0
        self.firstParams = []
        self.modelSolves = []
        self.gradSolves = []
        self._trueParams = trueParams
        self.evals = []
        self.bestParams = []
        self.bestCost = np.inf
        self.costTimes = []
        self.gradTimes = []

    def update(self, estimatedParams, cost=np.inf, incrementSolveCounter=True, solveType='cost', solveTime=np.NaN):
        """
        This method updates the performance metric tracking vectors with new values. It should only need to be called by a benchmarker class.

        Parameters
        ----------

        estimatedParams : list
            The vector of parameters that are being evaluated, after any transformations to return them to the original parameter space have been applied.
        cost : float, optional
            The RMSE cost of the parameter vectors that are being evaluated. The default is np.inf, to be used if parameters are out of bounds.
        incrementSolveCounter : bool, optional
            Should the solveCount be incremented, ie did the model need to be solved. This should only be False during benchmarker.evaluate() or if the parameters were out of bounds. The default is True.
        solveType : string, optional
            What type of model solve was used. 'grad' for with sensitivities and 'cost' for without.
        solveTime : float, optional
            A float representing the time, in seconds, it took to solve the model.

        Returns
        -------
        None.

        """
        if len(self.firstParams) == 0:
            self.firstParams = np.copy(estimatedParams)
        # Cast to numpy arrays
        trueParams = np.array(self._trueParams)
        estimatedParams = np.array(estimatedParams)
        # Update performance metrics
        self.paramRMSRE.append(np.sqrt(np.mean(((estimatedParams - trueParams) / trueParams) ** 2)))
        self.paramIdentifiedCount.append(np.sum(np.abs((estimatedParams - trueParams) / trueParams) < 0.05))
        self.costs.append(cost)
        if incrementSolveCounter:
            if solveType == 'cost':
                self.solveCount += 1
                self.costTimes.append(solveTime)
            elif solveType == 'grad':
                self.gradSolveCount += 1
                self.gradTimes.append(solveTime)
            self.check_repeated_param(estimatedParams, solveType)
            self.evals.append((estimatedParams, solveType))
        self.modelSolves.append(self.solveCount)
        self.gradSolves.append(self.gradSolveCount)
        if cost < self.bestCost:
            self.bestParams = estimatedParams
            self.bestCost = cost

    def plot(self):
        """
        This method plots the performance metrics as functions of time (in the order in which parameter vectors were evaluated). It will produce three plots, the RMSE cost, parameter RMSRE, and the number of identified parameters over the optimisation. This method will be called when benchmarker.evaluate() is called, so long as benchmarker.plotter = True (the default).

        Returns
        -------
        None.

        """
        # Cost plot
        plt.figure()
        plt.scatter(range(len(self.costs)), self.costs, c="k", marker=".")
        plt.xlabel('Model solves')
        plt.ylabel('RMSE cost')
        plt.title('Data error')

        # Parameter RMSRE plot
        plt.figure()
        plt.scatter(range(len(self.paramRMSRE)), self.paramRMSRE, c="k", marker=".")
        plt.xlabel('Model solves')
        plt.ylabel('Parameter RMSRE')
        plt.title('Parameter error')

        # Number of identified parameters plot
        plt.figure()
        plt.scatter(range(len(self.paramIdentifiedCount)), self.paramIdentifiedCount, c="k", marker=".")
        plt.xlabel('Model solves')
        plt.ylabel('Number of parameters identified')
        plt.title('Number of parameters identified')

        # Plot cost times
        if len(self.costTimes) > 0:
            plt.figure()
            n, _, _ = plt.hist(self.costTimes)
            plt.vlines(x=np.mean(self.costTimes), ymin=0, ymax=np.max(n), colors=['k'],
                       label=f'Mean: {np.mean(self.costTimes):.3f}')
            plt.xlabel('Time (sec)')
            plt.ylabel('Frequency')
            plt.title('Histogram of cost evaluation times')
            plt.legend()

        # Plot cost times
        if len(self.gradTimes) > 0:
            plt.figure()
            n, _, _ = plt.hist(self.gradTimes)
            plt.vlines(x=np.mean(self.gradTimes), ymin=0, ymax=np.max(n), colors=['k'],
                       label=f'Mean: {np.mean(self.gradTimes):.3f}')
            plt.xlabel('Time (sec)')
            plt.ylabel('Frequency')
            plt.title('Histogram of gradient evaluation times')
            plt.legend()

    def save(self, filename):
        """
        Saves the tracked variables. Useful to store results to plot later.

        Parameters
        ----------
        filename : string
            Filename for storing the tracked variables (solveCount, costs, modelSolves, paramRMSRE, paramIdentifiedCount). Variables will be pickled and stored in working directory.

        Returns
        -------
        None.

        """
        data = (self.solveCount, self.gradSolveCount, self.costs, self.modelSolves, self.gradSolves, self.paramRMSRE,
                self.paramIdentifiedCount, self.firstParams, self.evals, self.bestParams, self.bestCost, self.costTimes,
                self.gradTimes)
        with open(filename, 'wb') as f:
            pickle.dump(data, f)

    def load(self, filename):
        """
        Loads the tracked variables. Useful to store results to plot later.

        Parameters
        ----------
        filename : string
            Filename to load the stored tracked variables (solveCount, costs, modelSolves, paramRMSRE, paramIdentifiedCount). Variables will be read for the working directory. Must be saved using the .save() method associated with a tracker.

        Returns
        -------
        None.

        """
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        self.solveCount, self.gradSolveCount, self.costs, self.modelSolves, self.gradSolves, self.paramRMSRE, self.paramIdentifiedCount, self.firstParams, self.evals, self.bestParams, self.bestCost, self.costTimes, self.gradTimes = data

    def report_convergence(self):
        """
        Reports the performance metrics at the point of convergence, defined as the first point where there is an unbroken chain in the number of correctly identified parameters.

        Returns
        -------
        None.

        """
        finalParamId = self.paramIdentifiedCount[-1]
        ifEqualFinalParamId = self.paramIdentifiedCount == finalParamId
        ind = [i for i, x in enumerate(ifEqualFinalParamId) if
               x]  # Indexes where number of parameters identified is equal to the final count
        for i in ind:
            if all(ifEqualFinalParamId[i:]):
                # All future points remain with this many parameters identified, therefore it is considered converged
                print('Cost evaluations at convergence: ' + str(self.modelSolves[i]))
                print('Grad evaluations at convergence: ' + str(self.gradSolves[i]))
                print('Cost at convergence:             {0:.6f}'.format(self.costs[i]))
                print('Parameter RMSRE at convergence:  {0:.6f}'.format(self.paramRMSRE[i]))
                break

    def check_repeated_param(self, param, solveType):
        """
        Checks if a parameter vector has been evaluated before and reports a warning if it has.

        Parameters
        ----------
        param : array
            The parameters to check if the model has been already solved for.
        solveType : string
            Specifies the type of solve that param corresponds to. 'grad' refers to solving with sensitivities and 'cost' refers to one without. The allows check_repeated_params to ignore any cases where the parameters are first solved as a cost and then again as a grad which is common, expected and optimal for line search based gradient descent methods.

        Returns
        -------
        None.

        """
        for (p, st) in self.evals:
            if all(p == param):
                if st == solveType:
                    warnings.warn(
                        f'Duplicate solve. Both as {st}. This means the implementation of this optimiser can to be improved and the number of function evaluations can be reduced.')
                elif st == 'grad' and solveType == 'cost':
                    warnings.warn(
                        f'Duplicate solve. First as {st}, then as {solveType}. Cost can be found for free from a gradient solve. This means the implementation of this optimiser can to be improved and the number of function evaluations can be reduced.')


class Benchmarker:
    """
    The Benchmarker class contains all the features needed to evaluate an optimisation algorithm. This class should not need to be called directly and is instead used as a parent class for the benchmarker problems.

    The main methods to use from this class are n_parameters(), cost(), reset(), and evaluate().
    """

    def __init__(self):
        self._useScaleFactors = False
        self._parameters_bounded = False  # Should the parameters be bounded
        self._rates_bounded = False  # Should the rates be bounded
        self._logTransformParams = [False] * self.n_parameters()  # Are any of the parameter log-transformed
        self.plotter = True  # Should the performance metrics be plotted when evaluate() is called
        self.tracker = Tracker(self.defaultParams)  # Tracks the performance metrics
        if not hasattr(self, 'data'):
            self.data = None
        if not hasattr(self, 'lb'):
            self.lb = None
        if not hasattr(self, 'ub'):
            self.ub = None
        if not hasattr(self, 'simSens'):
            self.simSens = None
        if not hasattr(self, 'sensitivityCalc'):
            self.sensitivityCalc = None
        if not hasattr(self, 'rateMin'):
            self.rateMin = None
        if not hasattr(self, 'rateMax'):
            self.rateMax = None
        if not hasattr(self, 'vLow'):
            self.vLow = None
        if not hasattr(self, 'vHigh'):
            self.vHigh = None

    def load_data(self, dataPath=''):
        """
        Loads output data to use in fitting.

        Parameters
        ----------
        dataPath : string, optional
            An absolute filepath to the .csv data file. The default is '', in which case no file will be loaded.

        Returns
        -------
        None.

        """
        if not dataPath == '':
            tmp = []
            with open(dataPath, newline='') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    tmp.append(float(row[0]))
            self.data = np.array(tmp)

    def add_parameter_bounds(self, bounds, parameterSpace='original'):
        """
        Add bounds to the parameters. The bounds will be checked whenever the model is about to be solved, if they are violated then the model will not be solved and an infinite cost will be reported. The model solve count will not be incremented if the bounds are violated but the parameter vector will still be tracked for the other metrics.

        The bounds are checked after reversing any log-transforms on parameters (ie on exp(inputted parameters)).

        Parameters
        ----------
        bounds : list
            A list of bounds. The list should contain two elements, the first is a list of lower bounds, the same length as a parameter vector and the second a similar list of upper bounds. Use -np.inf or np.inf to not include bounds on particular parameters.
        parameterSpace : string
            Specify the parameter space of the inputted bounds. The options are 'original', to specify the bounds in the original parameter space, using the parameters that will be evaluated in the model, or 'input' for the parameter space inputted by the cost function, meaning they will be scaled by the default parameters if benchmarker._useScaleFactors=True, and log transformed if any parameters are to be log transformed. This will only have a difference if parameters are log transformed or if  benchmarker._useScaleFactors=True. The default is 'original'.

        Returns
        -------
        None.

        """
        if parameterSpace.lower() == 'input':
            self.lb = self.original_parameter_space(bounds[0])
            self.ub = self.original_parameter_space(bounds[1])
        elif parameterSpace.lower() == 'original':
            self.lb = bounds[0]
            self.ub = bounds[1]
        self._parameters_bounded = True

    def add_rate_bounds(self):
        """
        Add bounds to the rates. The bounds will be checked whenever the model is about to be solved, if they are violated then the model will not be solved and an infinite cost will be reported. The model solve count will not be incremented if the bounds are violated but the parameter vector will still be tracked for the other metrics.

        The bounds are checked after reversing any log-transforms on parameters (ie on exp(inputted parameters)).

        Parameters
        ----------
        None.
        Returns
        -------
        None.

        """
        self._rates_bounded = True

    def clamp_parameters(self, parameters):
        """
        Clamp a parameter vector to be between the current benchmarker bounds. Parameters should be specified in input parameter space. If no bounds are active or the inputted parameter vector obeys the current bounds, the inputted parameter vector is returned.

        Parameters
        ----------
        parameters : list
            Parameter vector in input space to clamp to the current bounds.

        Returns
        -------
        clampedParameters : list
            Parameter vector where the parameters that are out of bounds are clamped to the bounds.

        """
        p = np.copy(parameters)
        if not self.in_parameter_bounds(parameters):
            lb = self.input_parameter_space(self.lb)
            ub = self.input_parameter_space(self.ub)
            for i in range(self.n_parameters()):
                if p[i] < lb[i]:
                    p[i] = lb[i]
                elif p[i] > ub[i]:
                    p[i] = ub[i]
        return p

    def log_transform(self, whichParams=None):
        """
        Fit some parameters in a log-transformed space.

        Inputted log-transformed parameters will be set to exp(inputted parameters) before solving the model.

        Parameters
        ----------
        whichParams : list, optional
            Which parameters should be log-transformed, in the form of a list of booleans, the same length as the number of parameters, where True is a parameter to be log-transformed. The default is None, in which case all parameters will be log-transformed.

        Returns
        -------
        None.

        """
        if whichParams is None:  # Log-transform all parameters
            whichParams = [True] * self.n_parameters()
        self._logTransformParams = whichParams

    def input_parameter_space(self, parameters):
        """
        Maps parameters from the original parameter space to the input space. Incorporating any log transforms or scaling factors.

        Parameters
        ----------
        parameters : list
            Parameter vector in the original parameter space.

        Returns
        -------
        parameters : list
            Parameter vector mapped to input space.

        """
        parameters = np.copy(parameters)
        for i in range(self.n_parameters()):
            if self._useScaleFactors:
                parameters[i] = parameters[i] / self.defaultParams[i]
            if self._logTransformParams[i]:
                parameters[i] = np.log(parameters[i])

        return parameters

    def original_parameter_space(self, parameters):
        """
        Maps parameters from input space to the original parameter space. Removing any log transforms or scaling factors.

        Parameters
        ----------
        parameters : list
            Parameter vector in input space.

        Returns
        -------
        parameters : list
            Parameter vector mapped to the original parameter space.

        """
        parameters = np.copy(parameters)
        for i in range(self.n_parameters()):
            if self._logTransformParams[i]:
                parameters[i] = np.exp(parameters[i])
            if self._useScaleFactors:
                parameters[i] = parameters[i] * self.defaultParams[i]

        return parameters

    def transform_jacobian(self, parameters):
        """
        Finds the jacobian for the current parameter transform for derivatives calculated in original parameter space to be mapped to the input parameter space.

        Parameters
        ----------
        parameters : array
            Parameter vector at which the jacobian should be calculated. Given in input parameter space.

        Returns
        -------
        derivs : array
            Vector of transform derivatives. To map a derivative that was calculated in original parameter space to one calculated in input parameter space, it should be multiplied element-wise by derivs.

        """
        derivs = np.ones(self.n_parameters())
        for i in range(self.n_parameters()):
            if self._logTransformParams[i]:
                derivs[i] *= np.exp(parameters[i])
            if self._useScaleFactors:
                derivs[i] *= self.defaultParams[i]

        return derivs

    def in_parameter_bounds(self, parameters, boundedCheck=True):
        """
        Checks if parameters are inside any rectangular parameter bounds. If boundedCheck is True, then it will always return True if benchmarker._parameters_bounded = False.

        Parameters
        ----------
        parameters : list or numpy array
            Vector of parameters (in original parameter space) to check against the bounds.
        boundedCheck : bool, optional
            If True, and benchmarker._parameters_bounded=False, in_parameter_bounds will always return True. If False, it will ignore the value of bm._parameters_bounded and base the returned value only on the parameters and parameter bounds. The default is True.

        Returns
        -------
        bool
            True if parameters are inside the bounds or no bounds are specified, False if the parameters are outside the bounds.

        """
        if not self._parameters_bounded and boundedCheck:
            return True
        return self.parameter_penalty(parameters) == 0

    def in_rate_bounds(self, parameters, boundedCheck=True):
        """
        Checks if rates are inside bounds. If boundedCheck is True, then it will always return True if benchmarker._parameters_bounded = False.

        Parameters
        ----------
        parameters : list or numpy array
            Vector of parameters (in original parameter space) to check against the bounds.
        boundedCheck : bool, optional
            If True, and benchmarker._parameters_bounded=False, in_rate_bounds will always return True. If False, it will ignore the value of bm._parameters_bounded and base the returned value only on the parameters and rate bounds. The default is True.

        Returns
        -------
        bool
            True if rates are inside the bounds or no bounds are specified, False if the rates are outside the bounds.

        """
        if not self._rates_bounded and boundedCheck:
            return True
        return self.rate_penalty(parameters) == 0

    def n_parameters(self):
        """
        Returns the number of parameters in the model.
        """
        return len(self.defaultParams)

    def reset(self, fullReset=True):
        """
        Resets the benchmarker. This clears the simulation object and restarts the performance tracker. Optionally, the transforms and bounds can be turned off.

        Parameters
        ----------
        fullReset : bool, optional
            If True, transforms and bounds will be reset (turned off). The default is True.

        Returns
        -------
        None.

        """
        self.sim.reset()
        if self.sensitivityCalc:
            self.simSens.reset()
        self.tracker = Tracker(self.defaultParams)
        if fullReset:
            self.log_transform([False] * self.n_parameters())
            self._useScaleFactors = False
            self._parameters_bounded = False
            self._rates_bounded = False

    def cost(self, parameters, incrementSolveCounter=True):
        """
        Find the RMSE cost between the model solved using the inputted parameters and the data.

        Parameters
        ----------
        parameters : list or numpy array
            Vector of parameters to solve the model.
        incrementSolveCounter : bool, optional
            If False, it disables the solve counter tracker. This never needs to be set to False by a user. This is only required by the evaluate() method. The default is True.

        Returns
        -------
        cost : float
            The RMSE cost of the parameters.

        """
        testOutput = np.array(
            self.simulate(parameters, np.arange(0, self.tmax, self.freq), incrementSolveCounter=incrementSolveCounter))
        cost = self.rmse(testOutput, self.data)
        return cost

    def signed_error(self, parameters):
        """
        Similar to the cost method, but instead returns the vector of residuals/errors in the model output.

        Parameters
        ----------
        parameters : list or numpy array
            Vector of parameters to solve the model.

        Returns
        -------
        signed_error : numpy array
            The vector of model errors/residuals.

        """
        # Calculate cost for a given set of parameters
        testOutput = np.array(self.simulate(parameters, np.arange(0, self.tmax, self.freq)))
        return testOutput - self.data

    def squared_error(self, parameters):
        """
        Similar to the cost method, but instead returns the vector of squared residuals/errors in the model output.

        Parameters
        ----------
        parameters : list or numpy array
            Vector of parameters to solve the model.

        Returns
        -------
        signed_error : numpy array
            The vector of model squared errors/residuals.

        """
        # Calculate cost for a given set of parameters
        testOutput = np.array(self.simulate(parameters, np.arange(0, self.tmax, self.freq)))
        return (testOutput - self.data) ** 2

    def use_sensitivities(self):
        """
        Turn on sensitivities for a benchmarker. Also sets the parameter self.sensitivityCalc to True

        Parameters
        ----------
        None.

        Returns
        -------
        None.

        """
        paramNames = [self._paramContainer + '.p' + str(i + 1) for i in range(self.n_parameters())]
        self.simSens = myokit.Simulation(self.model, protocol=self.protocol(),
                                         sensitivities=([self._outputName], paramNames))
        self.simSens.set_tolerance(1e-9, 1e-9)
        self.sensitivityCalc = True

    def grad(self, parameters, incrementSolveCounter=True, inInputSpace=True, returnCost=False, residuals=False):
        """
        Find the gradient of the RMSE cost at the inputted parameters. Gradient is calculated using Myokit sensitivities.

        Parameters
        ----------
        parameters : list or numpy array
            Vector of parameters to find the derivatives about.
        incrementSolveCounter : bool, optional
            If False, it disables the solve counter tracker. This never needs to be set to False by a user. This is only required by the evaluate() method. The default is True.
        inInputSpace : bool, optional
            Specifies whether the inputted parameters are in input space. If True, then the derivative will be transformed into input space as well. The default is True.
        returnCost : bool, optional
            Whether the function should return the cost at parameters, in addition to the gradient. If True, it will return a tuple (cost, grad). If residuals is True, it will return (signed_error, grad). The default is False.
        residuals : bool, optional
            Whether the function should calculate the gradient of the cost, or the jacobian of the residual vector. If True, jacobian of residuals is returned instead of the gradient of the cost function. The default is False.

        Returns
        -------
        grad : array
            The gradient of the RMSE cost, evaluated at the inputted parameters.

        """
        curr = None
        sens = None
        J = None
        grad = None
        cost = None
        failed = False
        # Undo any transforms
        if inInputSpace:
            parameters = self.original_parameter_space(parameters)
        else:
            parameters = np.copy(parameters)

        # Check model is set up to solve for sensitivities
        if not self.sensitivityCalc:
            warnings.warn(
                "Current benchmarker problem not configured to use derivatives. Will recompile the simulation object with this enabled.")
            self.use_sensitivities()

        # Abort solving if the parameters are out of bounds
        if not self.in_parameter_bounds(parameters):
            failed = True
            incrementSolveCounter = False
            warnings.warn(
                'Tried to evaluate gradient when out of bounds. ionBench will try to resolve this by assuming infinite cost and a gradient that points back towards good parameters.')
            # The start and end variables won't be recorded since incrementSolveCounter is False but need to be defined for the tracker.update function to be called
            start = 0
            end = 0

        else:
            # Get sensitivities
            self.simSens.reset()
            self.set_params(parameters)
            self.set_steady_state(parameters)
            start = time.monotonic()
            try:
                curr, sens = self.solve_with_sensitivities(times=np.arange(0, self.tmax, self.freq))
                sens = np.array(sens)
            except myokit.SimulationError:
                failed = True
                warnings.warn(
                    'Tried to evaluate gradient but model failed to solve, likely poor choice of parameters. ionBench will try to resolve this by assuming infinite cost and a gradient that points back towards good parameters.')
            end = time.monotonic()

        if failed:
            self.tracker.update(parameters, incrementSolveCounter=incrementSolveCounter, solveType='grad',
                                solveTime=end - start)
            error = np.array([np.inf] * len(self.data))
            # use grad to point back to reasonable parameter space
            grad = -1 / (self.original_parameter_space(self.sample()) - parameters)
            if residuals:
                J = np.zeros((len(error), self.n_parameters()))
                for i in range(len(error)):
                    J[i,] = grad
        else:
            # Convert to cost derivative or residual jacobian
            error = curr - self.data
            cost = self.rmse(curr, self.data)

            self.tracker.update(parameters, cost=cost, incrementSolveCounter=incrementSolveCounter, solveType='grad',
                                solveTime=end - start)

            if residuals:
                J = np.zeros((len(curr), self.n_parameters()))
                for i in range(len(curr)):
                    for j in range(self.n_parameters()):
                        J[i, j] = sens[i, 0, j]
            else:
                grad = []
                for i in range(self.n_parameters()):
                    if 'moreno' in self._name:
                        grad.append(np.dot(error * self.weights, sens[:, 0, i]) / cost)
                    else:
                        grad.append(np.dot(error, sens[:, 0, i]) / (len(error) * cost))

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

    def set_params(self, parameters):
        """
        Set the parameters in the simulation object (both normal and sensitivity if needed). Inputted parameters should be in the original parameter space.

        Parameters
        ----------
        parameters : list or numpy array
            Vector of parameters to solve the model.

        Returns
        -------
        None.

        """
        # Update the parameters
        for i in range(self.n_parameters()):
            self.sim.set_constant(self._paramContainer + '.p' + str(i + 1), parameters[i])
            if self.sensitivityCalc:
                self.simSens.set_constant(self._paramContainer + '.p' + str(i + 1), parameters[i])
        # Workaround for myokit bug
        if 'moreno' in self._name:
            self.sim.set_parameters(self.sim.parameters())

    # noinspection PyProtectedMember
    def set_steady_state(self, parameters):
        """
        Sets the model to steady state at -80mV for the specified parameters. Will update the sensitivity simulation if self.sensitivityCalc == True.

        Parameters
        ----------
        parameters : list
            Vector of parameters to use for the calculation of the steady state.

        Returns
        -------
        None.

        """
        if 'moreno' in self._name:
            V = -120
        else:
            V = -80

        if 'hh' in str(type(self._analyticalModel)):
            t = [mg.tensor(a, dtype=np.double) for a in parameters]
            state = self._analyticalModel._steady_state_function(V, *t)
            s_state = np.zeros((len(parameters), len(state)))
            for j in range(len(state)):
                state[j].backward()
                sens = [float(t[i].grad) if t[i].grad is not None else 0 for i in range(self.n_parameters())]
                [t[i].null_grad() for i in range(self.n_parameters())]
                s_state[:, j] = sens

            state = [float(state[i]) for i in range(len(state))]
            s_state = [list(s_state[i]) for i in range(self.n_parameters())]
        else:
            assert 'markov' in str(type(self._analyticalModel))
            n = len(self._analyticalModel._states)
            f = ionbench.utils.autodiff.get_matrix_function(self._analyticalModel)

            state = np.zeros((len(parameters), n))
            s_state = np.zeros((len(parameters), n))
            for j in range(n):
                t = [mg.tensor(a, np.double) for a in parameters]
                At, _ = f(*t, V)
                B = At[:-1, -1:]
                A = At[:-1, :-1] - B

                xMG = ionbench.utils.autodiff.linalg_solve(A, -B)
                state = mg.zeros(n)
                for i in range(len(xMG)):
                    state[i] = xMG[i]
                state[-1] = 1 - np.sum(xMG)
                state[j].backward()
                sens = [float(t[i].grad) if t[i].grad is not None else 0 for i in range(self.n_parameters())]
                s_state[:, j] = sens
            state = [float(state[i]) for i in range(len(state))]
            s_state = [list(s_state[i]) for i in range(self.n_parameters())]
        # Check our steady state calculations don't differ from myokit
        try:
            myokitState = self._analyticalModel.steady_state(V, parameters)
            if np.linalg.norm(np.subtract(myokitState, state)) > 1e-6:
                raise RuntimeError(
                    f'Steady state calculated by ionBench differed from myokit by {np.linalg.norm(myokitState - state)}. Needs looking into to see if either myokit has changed steady states or the ionBench matrix solver is failing somewhere.')
        except myokit.lib.markov.LinearModelError as e:
            if 'eigenvalues' in str(e):
                warnings.warn(
                    'Positive eigenvalues found so steady state is unstable. Will use states defined in problem instead.')
                state = None
            else:
                raise
        if state is not None:
            self.sim.set_state(state)
            if self.sensitivityCalc:
                self.simSens.set_state(state)
                self.simSens._s_state = s_state

    def solve_with_sensitivities(self, times):
        """
        Solve the model with sensitivities to find both the current, curr, and the sensitivities dcurr/dp_i, sens.
        ----------
        times : list
            The times at which to record the current and sensitivities.

        Returns
        -------
        curr : numpy array
            The model outputted current.
        sens : numpy array
            The sensitivities of the outputted current.
        """
        log, e = self.simSens.run(self.tmax + 1, log_times=times)
        return np.array(log[self._outputName]), e

    def solve_model(self, times, continueOnError=True):
        """
        Solve the model at the inputted times and return the current trace.

        Parameters
        ----------
        times : list or numpy array
            Vector of time points to record model output. Typically, in ms.
        continueOnError : bool, optional
            If continueOnError is True, any errors that occur during solving the model will be ignored and an infinite output will be given. The default is True.

        Returns
        -------
        modelOutput : list
            A vector of model outputs (current trace).

        """
        if continueOnError:
            try:
                log = self.sim.run(self.tmax + 1, log_times=times)
                return np.array(log[self._outputName])
            except myokit.SimulationError:
                warnings.warn("Failed to solve model. Will report infinite output in the hope of continuing the run.")
                return np.array([np.inf] * len(times))
        else:
            log = self.sim.run(self.tmax + 1, log_times=times)
            return np.array(log[self._outputName])

    def simulate(self, parameters, times, continueOnError=True, incrementSolveCounter=True):
        """
        Simulate the model for the inputted parameters and return the model output at the specified times.

        Parameters
        ----------
        parameters : list or numpy array
            Vector of parameters to solve the model.
        times : list or numpy array
            Vector of time points to record model output. Typically, in ms.
        continueOnError : bool, optional
            If continueOnError is True, any errors that occur during solving the model will be ignored and an infinite output will be given. The default is True.
        incrementSolveCounter : bool, optional
            If False, it disables the solve counter tracker. This never needs to be set to False by a user. This is only required by the evaluate() method. The default is True.

        Returns
        -------
        modelOutput : list
            A vector of model outputs (current trace).

        """
        # Return the parameters to the original parameter space
        parameters = self.original_parameter_space(parameters)  # Creates a copy of the parameter vector

        # Reset the simulation
        self.sim.reset()

        # Abort solving if the parameters are out of bounds
        penalty = 0
        if self._parameters_bounded or 'staircase' in self._name:
            penalty += self.parameter_penalty(parameters)
        if self._rates_bounded or 'staircase' in self._name:
            penalty += self.rate_penalty(parameters)
        if penalty > 0:
            self.tracker.update(parameters, cost=penalty, incrementSolveCounter=False)
            return np.add(penalty, self.data)

        # Set the parameters in the simulation object
        self.set_params(parameters)
        self.set_steady_state(parameters)

        # Run the simulation and track the performance
        start = time.monotonic()
        out = self.solve_model(times, continueOnError=continueOnError)
        end = time.monotonic()
        self.tracker.update(parameters, cost=self.rmse(out, self.data), incrementSolveCounter=incrementSolveCounter,
                            solveTime=end - start)
        return out

    def parameter_penalty(self, parameters):
        """
        Penalty function for out of bound parameters. The penalty for each parameter p that is out of bounds is 1e4+1e4*|p-bound| where bound is either the upper or lower bound for p, whichever was violated.
        Parameters
        ----------
        parameters : numpy array
            A possibly out-of-bounds parameter vector to penalise.
        Returns
        -------
        penalty : float
            The penalty to apply for parameter bound violations.
        """
        # Penalty increases linearly with parameters out of bounds
        penalty = 1e4 * np.sum(np.abs(parameters - self.ub), where=parameters > self.ub)
        penalty += 1e4 * np.sum(np.abs(parameters - self.lb), where=parameters < self.lb)
        # Minimum penalty per parameter violation
        penalty += 1e4 * np.sum(np.logical_or(parameters > self.ub, parameters < self.lb))
        return penalty

    def rate_penalty(self, parameters):
        """
        Penalty function for out of bound rates. The penalty for each rate r that is out of bounds is 1e4+1e4*|r-bound| where bound is either the upper or lower bound for r, whichever was violated.
        Parameters
        ----------
        parameters : numpy array
            A possibly out-of-bounds parameter vector to penalise.
        Returns
        -------
        penalty : float
            The penalty to apply for rate bound violations.
        """
        penalty = 0
        for rateFunc, rateType in self._rateFunctions:
            if rateType == 'positive':
                # check kHigh is in bounds
                k = rateFunc(parameters, self.vHigh)
            elif rateType == 'negative':
                # check kLow is in bounds
                k = rateFunc(parameters, self.vLow)
            elif rateType == 'independent':
                # check rate in bounds
                k = rateFunc(parameters, 0)  # Voltage doesn't matter
            else:
                raise RuntimeError(
                    "Error in bm._rateFunctions. Doesn't contain 'positive', 'negative', or 'independent' in at least one position. Check for typos.")
            if k < self.rateMin:
                penalty += 1e4
                penalty += 1e4 * np.abs(k - self.rateMin)
            elif k > self.rateMax:
                penalty += 1e4
                penalty += 1e4 * np.abs(k - self.rateMax)
        return penalty

    def rmse(self, c1, c2):
        """
        Returns the RMSE between c1 and c2. This function is overridden in the Moreno problem to do weighted RMSE.

        Parameters
        ----------
        c1 : numpy array
            A list of model outputs, typically current.
        c2 : numpy array
            The data to compare the model output to. Should be the same size as c1.

        Returns
        -------
        rmse : float
            The RMSE between c1 and c2.
        """
        return np.sqrt(np.mean((c1 - c2) ** 2))

    def evaluate(self, parameters):
        """
        Evaluates a final set of parameters.

        This method reports the performance metrics for this parameter vector (calling evaluate() does NOT increase the number of cost function evaluations). If benchmarker.plotter = True, then it also plots the performance metrics over the course of the optimisation.

        Parameters
        ----------
        parameters : list or numpy array
            Vector of parameters to solve the model, evaluating the performance of the final parameter vector.

        Returns
        -------
        None.

        """
        print('')
        print('=========================================')
        print('===    Evaluating Final Parameters    ===')
        print('=========================================')
        print('')
        print('Number of cost evaluations:      ' + str(self.tracker.solveCount))
        print('Number of grad evaluations:      ' + str(self.tracker.gradSolveCount))
        cost = self.cost(parameters, incrementSolveCounter=False)
        print('Final cost:                      {0:.6f}'.format(cost))
        print('Parameter RMSRE:                 {0:.6f}'.format(self.tracker.paramRMSRE[-1]))
        print('Number of identified parameters: ' + str(self.tracker.paramIdentifiedCount[-1]))
        print('Total number of parameters:      ' + str(self.n_parameters()))
        print('Best cost:                       {0:.6f}'.format(self.tracker.bestCost))
        self.tracker.report_convergence()
        print('')
        if self.plotter:
            self.tracker.plot()
            self.sim.reset()
            self.set_params(self.tracker.firstParams)
            self.set_steady_state(self.tracker.firstParams)
            firstOut = self.solve_model(np.arange(0, self.tmax, self.freq), continueOnError=True)
            self.sim.reset()
            self.set_params(self.original_parameter_space(parameters))
            self.set_steady_state(self.original_parameter_space(parameters))
            lastOut = self.solve_model(np.arange(0, self.tmax, self.freq), continueOnError=True)
            plt.figure()
            if "moreno" in self._name:
                plt.plot(self.data)
                plt.plot(firstOut)
                plt.plot(lastOut)
                plt.ylabel('Summary Statistics')
            else:
                plt.plot(np.arange(0, self.tmax, self.freq), self.data)
                plt.plot(np.arange(0, self.tmax, self.freq), firstOut)
                plt.plot(np.arange(0, self.tmax, self.freq), lastOut)
                plt.ylabel('Current')
                plt.xlabel('Time (ms)')
            plt.legend(['Data', 'First Parameters', 'Final Parameters'])
            plt.title('Improvement after fitting: ' + self._name)
            plt.show()

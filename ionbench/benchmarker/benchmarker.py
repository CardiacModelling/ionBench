"""
Contains the main Benchmarker class from which the benchmarker problems inherit the vast majority of their methods.
Contains the Tracker class which is stored as an attribute in each benchmarker and managers tracking the performance metrics over an optimisation.
"""
import numpy as np
import myokit
import myokit.lib.markov
import csv
import ionbench
import matplotlib.pyplot as plt
import warnings
import time
import mygrad as mg

from ionbench.tracker import Tracker


class Benchmarker:
    """
    The Benchmarker class contains all the features needed to evaluate an optimisation algorithm. This class should not need to be called directly and is instead used as a parent class for the benchmarker problems.

    The main methods to use from this class are n_parameters(), cost(), reset(), and evaluate().
    """

    def __init__(self):
        # Ensure any undefined attributes (those defined by subclasses) are defined if missing
        if not hasattr(self, 'DATA'):  # pragma: no cover
            self.DATA = np.array(None)
        if not hasattr(self, '_LOWER_BOUND'):  # pragma: no cover
            self._LOWER_BOUND = np.array(None)
        if not hasattr(self, '_UPPER_BOUND'):  # pragma: no cover
            self._UPPER_BOUND = np.array(None)
        if not hasattr(self, 'lb'):  # pragma: no cover
            self.lb = np.array(None)
        if not hasattr(self, 'ub'):  # pragma: no cover
            self.ub = np.array(None)
        if not hasattr(self, 'simSens'):  # pragma: no cover
            self.simSens = None
        if not hasattr(self, 'sensitivityCalc'):  # pragma: no cover
            self.sensitivityCalc = None
        if not hasattr(self, 'RATE_MIN'):  # pragma: no cover
            self.RATE_MIN = None
        if not hasattr(self, 'RATE_MAX'):  # pragma: no cover
            self.RATE_MAX = None
        if not hasattr(self, 'COST_THRESHOLD'):  # pragma: no cover
            self.COST_THRESHOLD = None
        if not hasattr(self, '_TRUE_PARAMETERS'):  # pragma: no cover
            self._TRUE_PARAMETERS = np.array(None)
        if not hasattr(self, 'sim'):  # pragma: no cover
            self.sim = None
        if not hasattr(self, 'T_MAX'):  # pragma: no cover
            self.T_MAX = None
        if not hasattr(self, 'TIMESTEP'):  # pragma: no cover
            self.TIMESTEP = None
        if not hasattr(self, '_PARAMETER_CONTAINER'):  # pragma: no cover
            self._PARAMETER_CONTAINER = ''
        if not hasattr(self, '_OUTPUT_NAME'):  # pragma: no cover
            self._OUTPUT_NAME = ''
        if not hasattr(self, '_TOLERANCES'):  # pragma: no cover
            self._TOLERANCES = None
        if not hasattr(self, 'NAME'):  # pragma: no cover
            self.NAME = ''
        if not hasattr(self, '_MODEL'):  # pragma: no cover
            self._MODEL = None
        if not hasattr(self, 'WEIGHTS'):  # pragma: no cover
            self.WEIGHTS = None
        if not hasattr(self, '_ANALYTICAL_MODEL'):  # pragma: no cover
            self._ANALYTICAL_MODEL = None
        if not hasattr(self, '_RATE_FUNCTIONS'):  # pragma: no cover
            self._RATE_FUNCTIONS = None

        self.useScaleFactors = False
        self.logTransformParams = [False] * self.n_parameters()  # Are any of the parameter log-transformed
        if 'staircase' in self.NAME:
            # Staircase problems should have bounds turned on
            self.add_parameter_bounds()
            self.add_rate_bounds()
        else:
            # Otherwise they start turned off
            self.parametersBounded = False
            self.ratesBounded = False
        self.plotter = True  # Should the performance metrics be plotted when evaluate() is called
        self.tracker = Tracker()  # Tracks the performance metrics
        self.V_LOW = -120
        self.V_HIGH = 60

        # Set numpy writeable flags
        self._TRUE_PARAMETERS.flags['WRITEABLE'] = False
        self._LOWER_BOUND.flags['WRITEABLE'] = False
        self._UPPER_BOUND.flags['WRITEABLE'] = False

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
        # Load the current/output data from ionBench
        if not dataPath == '':
            tmp = []
            with open(dataPath, newline='') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    tmp.append(float(row[0]))
            self.DATA = np.array(tmp)

    def sample(self, n=1):  # pragma: no cover
        raise NotImplementedError

    @staticmethod
    def protocol():  # pragma: no cover
        raise NotImplementedError

    def add_parameter_bounds(self):
        """
        Add bounds to the parameters. The bounds will be checked whenever the model is about to be solved, if they are violated then the model will not be solved and a penalty cost will be reported. The model solve count will not be incremented if the bounds are violated but the parameter vector will still be tracked for the other metrics.
        """
        self.lb = np.copy(self._LOWER_BOUND)
        self.ub = np.copy(self._UPPER_BOUND)
        self.parametersBounded = True

    def add_rate_bounds(self):
        """
        Add bounds to the rates. The bounds will be checked whenever the model is about to be solved, if they are violated then the model will not be solved and an infinite cost will be reported. The model solve count will not be incremented if the bounds are violated but the parameter vector will still be tracked for the other metrics.

        The bounds are checked after reversing any log-transforms on parameters (ie on exp(inputted parameters)).
        """
        self.ratesBounded = True

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
        # Don't edit the inputted parameter vector
        p = np.copy(parameters)
        # If out of bounds
        if not self.in_parameter_bounds(parameters):
            # Find bounds in input space
            lb = self.input_parameter_space(self.lb)
            ub = self.input_parameter_space(self.ub)
            # Clamp to input space bounds
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
        self.logTransformParams = whichParams

    def input_parameter_space(self, parameters):
        """
        Maps parameters from the original parameter space to the input space. Incorporating any log transforms or scaling factors.

        Parameters
        ----------
        parameters : list
            Parameter vector in the original parameter space.

        Returns
        -------
        parameters : np.ndarray
            Parameter vector mapped to input space.

        """
        # Don't edit the inputted parameter vector
        parameters = np.copy(parameters)
        for i in range(self.n_parameters()):
            # Apply scale factor transforms
            if self.useScaleFactors:
                parameters[i] = parameters[i] / self._TRUE_PARAMETERS[i]
            # Apply log transforms
            if self.logTransformParams[i]:
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
        parameters : np.ndarray
            Parameter vector mapped to the original parameter space.

        """
        # Don't edit the inputted parameter vector
        parameters = np.copy(parameters)
        for i in range(self.n_parameters()):  # Note the reverse order compared with input_parameter_space
            # Remove log transforms
            if self.logTransformParams[i]:
                parameters[i] = np.exp(parameters[i])
            # Remove scale factor transform
            if self.useScaleFactors:
                parameters[i] = parameters[i] * self._TRUE_PARAMETERS[i]

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
        # Create transform vector of all 1s
        derivs = np.ones(self.n_parameters())
        # Apply transforms to derivs
        for i in range(self.n_parameters()):
            if self.logTransformParams[i]:
                derivs[i] *= np.exp(parameters[i])
            if self.useScaleFactors:
                derivs[i] *= self._TRUE_PARAMETERS[i]

        return derivs

    def in_parameter_bounds(self, parameters, boundedCheck=True):
        """
        Checks if parameters are inside any rectangular parameter bounds. If boundedCheck is True, then it will always return True if benchmarker.parametersBounded = False.

        Parameters
        ----------
        parameters : list or numpy array
            Vector of parameters (in original parameter space) to check against the bounds.
        boundedCheck : bool, optional
            If True, and benchmarker.parametersBounded=False, in_parameter_bounds will always return True. If False, it will ignore the value of bm.parametersBounded and base the returned value only on the parameters and parameter bounds. The default is True.

        Returns
        -------
        bool
            True if parameters are inside the bounds or no bounds are specified, False if the parameters are outside the bounds.

        """
        if not self.parametersBounded and boundedCheck:
            return True
        return self.parameter_penalty(parameters) == 0

    def in_rate_bounds(self, parameters, boundedCheck=True):
        """
        Checks if rates are inside bounds. If boundedCheck is True, then it will always return True if benchmarker.parametersBounded = False.

        Parameters
        ----------
        parameters : list or numpy array
            Vector of parameters (in original parameter space) to check against the bounds.
        boundedCheck : bool, optional
            If True, and benchmarker.parametersBounded=False, in_rate_bounds will always return True. If False, it will ignore the value of bm.parametersBounded and base the returned value only on the parameters and rate bounds. The default is True.

        Returns
        -------
        bool
            True if rates are inside the bounds or no bounds are specified, False if the rates are outside the bounds.

        """
        if not self.ratesBounded and boundedCheck:
            return True
        return self.rate_penalty(parameters) == 0

    def set_max_iter_flag(self, hitMaxIter=True):
        """
        Sets a flag in ionBench to indicate that the optimisation algorithm has hit the maximum number of iterations. This is used to note any optimisation runs that may need repeating, as we try to avoid terminating due to maximum number of iterations. This flag is also set for similar termination criteria like maximum number of function evaluations. The flag is stored in the Tracker.

        Parameters
        ----------
        hitMaxIter : bool, optional
        Set tracker.maxIterFlag to this value. The default is True.

        Returns
        -------
        None.
        """
        self.tracker.maxIterFlag = hitMaxIter

    def n_parameters(self):
        """
        Returns the number of parameters in the model.
        """
        return len(self._TRUE_PARAMETERS)

    def reset(self, fullReset=True):
        """
        Resets the benchmarker. This clears the simulation object and restarts the performance tracker. Optionally, the transforms and bounds can be turned off.

        Parameters
        ----------
        fullReset : bool, optional
            If True, transforms and bounds will be reset (turned off except staircase which will turn bounds on). The default is True.
        """
        # Reset the simulation object
        self.sim.reset()
        # If we have sensitivities enabled, reset that too
        if self.sensitivityCalc:
            self.simSens.reset()
        # Generate a new tracker
        self.tracker = Tracker()
        # Are we resetting the modification too?
        if fullReset:
            self.log_transform([False] * self.n_parameters())
            self.useScaleFactors = False
            # Keep staircase bounds consistent
            if 'staircase' in self.NAME:
                self.add_rate_bounds()
                self.add_parameter_bounds()
            else:
                self.parametersBounded = False
                self.ratesBounded = False
            self.lb = np.copy(self._LOWER_BOUND)
            self.ub = np.copy(self._UPPER_BOUND)
            self.RATE_MIN = 1.67e-5
            self.RATE_MAX = 1e3

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
            self.simulate(parameters, np.arange(0, self.T_MAX, self.TIMESTEP),
                          incrementSolveCounter=incrementSolveCounter))
        cost = self.rmse(testOutput, self.DATA)
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
        testOutput = np.array(self.simulate(parameters, np.arange(0, self.T_MAX, self.TIMESTEP)))
        return testOutput - self.DATA

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
        testOutput = np.array(self.simulate(parameters, np.arange(0, self.T_MAX, self.TIMESTEP)))
        return (testOutput - self.DATA) ** 2

    def use_sensitivities(self):
        """
        Turn on sensitivities for a benchmarker. Also sets the parameter self.sensitivityCalc to True. If sensitivities are already enabled (by checking bm.sensitivityCalc), then this method will do nothing.
        """
        # Check sensitivities aren't already enabled
        if not self.sensitivityCalc:
            # Enable sensitivities and compile
            paramNames = [self._PARAMETER_CONTAINER + '.p' + str(i + 1) for i in range(self.n_parameters())]
            self.simSens = myokit.Simulation(self._MODEL, protocol=self.protocol(),
                                             sensitivities=([self._OUTPUT_NAME], paramNames))
            self.simSens.set_tolerance(*self._TOLERANCES)
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
        # Now to find grad, cost, J (jacobian) and error (residuals)
        # First check if the parameters are out of bounds
        # Abort solving if the parameters are out of bounds
        inParameterBounds = self.in_parameter_bounds(parameters)
        inRateBounds = self.in_rate_bounds(parameters)
        if not inParameterBounds or not inRateBounds:
            # Gradient out of bounds so use penalty function
            # Calculate gradient of penalty function
            parametersMG = [mg.Tensor(p) for p in parameters]
            penalty = self.parameter_penalty(parametersMG) + self.rate_penalty(parametersMG)
            assert penalty.data > 0
            penalty.backward()
            grad = np.zeros(self.n_parameters())
            for i in range(self.n_parameters()):
                grad[i] = parametersMG[i].grad if parametersMG[i].grad is not None else 0
            cost = float(penalty.data)
            error = cost * np.ones(len(self.DATA))
            J = np.zeros((len(self.DATA), self.n_parameters()))
            for i in range(len(self.DATA)):
                J[i,] = grad
            self.tracker.update(parameters, incrementSolveCounter=False, solveType='grad', cost=cost,
                                solveTime=0)
        else:
            # Inside any bounds, so we can get sensitivities
            # Will get sensitivities and find the J (jacobian) and curr (current), then exit the try except else block to calculate grad, cost and error from J and curr
            self.simSens.reset()
            self.set_params(parameters)
            self.set_steady_state(parameters)
            start = time.monotonic()
            try:
                curr, sens = self.solve_with_sensitivities(times=np.arange(0, self.T_MAX, self.TIMESTEP))
                sens = np.array(sens)
            except (myokit.SimulationError, np.linalg.LinAlgError):  # pragma: no cover
                # If the model fails to solve, we will assume the cost is infinite and the jacobian/gradient points back towards good parameters
                end = time.monotonic()
                warnings.warn(
                    'Tried to evaluate gradient but model failed to solve, likely poor choice of parameters. ionBench will try to resolve this by assuming infinite cost and a gradient that points back towards good parameters.')
                self.tracker.update(parameters, incrementSolveCounter=False, solveType='grad',
                                    solveTime=end - start)
                curr = np.array([np.inf] * len(self.DATA))
                # use grad to point back to reasonable parameter space
                # Define jacobian
                J = np.zeros((len(curr), self.n_parameters()))
                tmp = -1 / (self.original_parameter_space(self.sample()) - parameters)
                for i in range(len(curr)):
                    J[i,] = tmp
            else:
                # Here the model successfully solved, so we can calculate the jacobian
                end = time.monotonic()
                # Define the jacobian from the sensitivities
                J = np.zeros((len(curr), self.n_parameters()))
                for i in range(len(curr)):
                    for j in range(self.n_parameters()):
                        J[i, j] = sens[i, 0, j]
            # Calculate the gradient from the jacobian
            grad = []
            error = curr - self.DATA
            cost = self.rmse(curr, self.DATA)
            for i in range(self.n_parameters()):
                if 'moreno' in self.NAME:
                    grad.append(np.dot(error * self.WEIGHTS, J[:, i]) / cost)
                else:
                    grad.append(np.dot(error, J[:, i]) / (len(self.DATA) * cost))
            self.tracker.update(parameters, cost=cost, incrementSolveCounter=incrementSolveCounter,
                                solveType='grad',
                                solveTime=end - start)
        return self.map_derivative(J, grad, parameters, inInputSpace, returnCost, residuals, cost, error)

    def map_derivative(self, J, grad, parameters, inInputSpace, returnCost, residuals, cost, error):
        """
        Map the derivatives (J/jacobian and grad/gradient) onto the correct parameter space and then return the specified outputs. Outputs are described in the docstring for grad.
        """
        # Map derivatives to input space
        if inInputSpace:
            derivs = self.transform_jacobian(self.input_parameter_space(parameters))
            if residuals:
                J *= derivs
            else:
                grad *= derivs
        # Handle return options
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
        """
        # Update the parameters
        for i in range(self.n_parameters()):
            self.sim.set_constant(self._PARAMETER_CONTAINER + '.p' + str(i + 1), parameters[i])
            if self.sensitivityCalc:
                self.simSens.set_constant(self._PARAMETER_CONTAINER + '.p' + str(i + 1), parameters[i])
        # Workaround for old myokit bug - https://github.com/myokit/myokit/issues/1044 - fixed in 1.36.0
        if 'moreno' in self.NAME:
            self.sim.set_parameters(self.sim.parameters())

    # noinspection PyProtectedMember
    def set_steady_state(self, parameters):
        """
        Sets the model to steady state at -80mV for the specified parameters. Will update the sensitivity simulation if self.sensitivityCalc == True.

        Parameters
        ----------
        parameters : list or numpy array
            Vector of parameters to use for the calculation of the steady state.
        """
        # Initial membrane voltage
        if 'moreno' in self.NAME:
            V = -120
        else:
            V = -80
        At = np.zeros(self.n_parameters())
        if 'hh' in str(type(self._ANALYTICAL_MODEL)):
            # HH does autodiff through myokit
            t = [mg.tensor(a, dtype=np.double) for a in parameters]
            state = self._ANALYTICAL_MODEL._steady_state_function(V, *t)
            s_state = np.zeros((len(parameters), len(state)))
            for j in range(len(state)):
                state[j].backward()
                sens = [float(t[i].grad) if t[i].grad is not None else 0 for i in range(self.n_parameters())]
                [t[i].null_grad() for i in range(self.n_parameters())]
                s_state[:, j] = sens

            state = [float(state[i]) for i in range(len(state))]
            s_state = [list(s_state[i]) for i in range(self.n_parameters())]
        else:
            # MM does autodiff through ionBench reproduction of some myokit code
            assert 'markov' in str(type(self._ANALYTICAL_MODEL))
            n = len(self._ANALYTICAL_MODEL._states)
            f = ionbench.utils.autodiff.get_matrix_function(self._ANALYTICAL_MODEL)

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
            myokitState = self._ANALYTICAL_MODEL.steady_state(V, parameters)
            if np.linalg.norm(np.subtract(myokitState, state)) > 1e-6:  # pragma: no cover
                warnings.warn(f'Myokit and ionBench seems to slightly (norm: {np.linalg.norm(myokitState - state)}) disagree on the steady state. Will use the myokit steady state.')
                state = myokitState
        except myokit.lib.markov.LinearModelError as e:  # pragma: no cover
            if 'eigenvalues' in str(e):
                warnings.warn(
                    'Positive eigenvalues found so steady state is unstable. Will use states defined in problem instead.')
                state = None
            else:
                raise
        except np.linalg.LinAlgError as e:  # pragma: no cover
            if 'infs or NaNs' in str(e):
                # Infs or NaNs in matrix, caused error in myokit steady state
                warnings.warn('Infs or NaNs in matrix so steady state is invalid. Will use states defined in problem instead.')
                state = None
            else:
                raise
        # Now the steady state has been calculated we check it makes sense
        if state is not None:
            if np.any(np.array(state) < 0):
                npstate = np.array(state)
                if np.all(np.abs(npstate[npstate < 0]) < 1e-12):
                    # If the negative values are very close to zero, then we can assume they are zero
                    state = [max(0, s) / np.sum(npstate) for s in npstate]
                else:  # pragma: no cover
                    # Some parameter combinations lead to negative steady state values (not just rounding errors). We don't set the state in this case as a positive steady state doesn't exist.
                    if np.any(At < 0):
                        warnings.warn(
                            'The parameters give negative transition probabilities which leads to an invalid steady state. Will use states defined in the problem instead.')
                        state = None
                    else:
                        print(parameters)
                        print(At)
                        warnings.warn(
                            "Steady state contains significant (>1e-12) negative values. If you see this, please open an issue on the GitHub repo. For now, I will continue with the myokits state (which hopefully doesn't have negative values)")
                        state = myokitState
        # Finally, set the initial state
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
        log, e = self.simSens.run(self.T_MAX + 1, log=[self._OUTPUT_NAME], log_times=times)
        return np.array(log[self._OUTPUT_NAME]), e

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
                if 'staircase' in self.NAME:
                    log = self.sim.run(self.T_MAX + 1, log=[self._OUTPUT_NAME], log_times=times)
                else:
                    log = self.sim.run(self.T_MAX + 1, log_times=times)
                return np.array(log[self._OUTPUT_NAME])
            except (myokit.SimulationError, np.linalg.LinAlgError):  # pragma: no cover
                warnings.warn("Failed to solve model. Will report infinite output in the hope of continuing the run.")
                return np.array([np.inf] * len(times))
        else:
            log = self.sim.run(self.T_MAX + 1, log_times=times)
            return np.array(log[self._OUTPUT_NAME])

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
        if self.parametersBounded:
            penalty += self.parameter_penalty(parameters)
        if self.ratesBounded:
            penalty += self.rate_penalty(parameters)
        if penalty > 0:
            self.tracker.update(parameters, cost=penalty, incrementSolveCounter=False)
            return np.add(penalty, self.DATA)

        # Set the parameters in the simulation object
        self.set_params(parameters)
        self.set_steady_state(parameters)

        # Run the simulation and track the performance
        start = time.monotonic()
        out = self.solve_model(times, continueOnError=continueOnError)
        end = time.monotonic()
        self.tracker.update(parameters, cost=self.rmse(out, self.DATA), incrementSolveCounter=incrementSolveCounter,
                            solveTime=end - start)
        return out

    def parameter_penalty(self, parameters):
        """
        Penalty function for out of bound parameters. The penalty for each parameter p that is out of bounds is 1e5+1e5*|p-bound| where bound is either the upper or lower bound for p, whichever was violated.

        Parameters
        ----------
        parameters : numpy array or mygrad.Tensor
            A possibly out-of-bounds parameter vector to penalise.

        Returns
        -------
        penalty : float or mygrad.Tensor
            The penalty to apply for parameter bound violations.
        """
        # Penalty increases linearly with parameters out of bounds
        penalty = 0
        for i in range(self.n_parameters()):
            if parameters[i] < self.lb[i]:
                penalty += 1e5 + 1e5 * np.log(1 + np.abs(parameters[i] - self.lb[i]))
            elif parameters[i] > self.ub[i]:
                penalty += 1e5 + 1e5 * np.log(1 + np.abs(parameters[i] - self.ub[i]))
        return penalty

    def rate_penalty(self, parameters):
        """
        Penalty function for out of bound rates. The penalty for each rate r that is out of bounds is 1e5+1e5*|r-bound| where bound is either the upper or lower bound for r, whichever was violated.

        Parameters
        ----------
        parameters : numpy array or mygrad.Tensor
            A possibly out-of-bounds parameter vector to penalise.

        Returns
        -------
        penalty : float or mygrad.Tensor
            The penalty to apply for rate bound violations.
        """
        penalty = 0
        # Loop though the problem-specific rate functions
        for i, rateFunc in enumerate(self._RATE_FUNCTIONS):
            # Is the rate maximised at high or low voltages
            k = max(rateFunc(parameters, self.V_HIGH), rateFunc(parameters, self.V_LOW))
            if 'moreno' in self.NAME:  # Moreno is naturally out of bounds for some rates so the bounds are wider there
                if k < self.RATE_MIN:
                    penalty += 1e5
                    penalty += 1e5 * np.log(1 + np.abs(k - self.RATE_MIN))
                elif (k > self.RATE_MAX and i not in [3, 5]) or (k > self.RATE_MAX * 10000 and i in [3, 5]):
                    penalty += 1e5
                    penalty += 1e5 * np.log(1 + np.abs(k - self.RATE_MAX))
            else:
                if k < self.RATE_MIN:
                    penalty += 1e5
                    penalty += 1e5 * np.log(1 + np.abs(k - self.RATE_MIN))
                elif k > self.RATE_MAX:
                    penalty += 1e5
                    penalty += 1e5 * np.log(1 + np.abs(k - self.RATE_MAX))
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

    def is_converged(self):
        """
        Returns whether the optimisation has converged yet (satisfied cost threshold or number of unchanged costs).

        Returns
        -------
        bool
            True if the optimisation has converged, False if it has not.
        """
        return self.tracker.cost_threshold(threshold=self.COST_THRESHOLD) or self.tracker.cost_unchanged()

    def evaluate(self):
        """
        Evaluates the best parameters.

        This method reports the performance metrics for the best parameter vector (calling evaluate() does NOT increase the number of cost function evaluations). If benchmarker.plotter = True, then it also plots the performance metrics over the course of the optimisation.
        """
        print('')
        print('========================================')
        print('===      Evaluating Performance      ===')
        print('========================================')
        print('')
        print('Number of cost evaluations:      ' + str(self.tracker.costSolveCount))
        print('Number of grad evaluations:      ' + str(self.tracker.gradSolveCount))
        print('Best cost:                       {0:.6f}'.format(self.tracker.bestCost))
        self.tracker.report_convergence(self.COST_THRESHOLD)
        print('')
        if self.plotter:
            try:
                self.tracker.plot()
                self.sim.reset()
                self.set_params(self.tracker.firstParams)
                self.set_steady_state(self.tracker.firstParams)
                firstOut = self.solve_model(np.arange(0, self.T_MAX, self.TIMESTEP), continueOnError=True)
                self.sim.reset()
                self.set_params(self.tracker.bestParams)
                self.set_steady_state(self.tracker.bestParams)
                lastOut = self.solve_model(np.arange(0, self.T_MAX, self.TIMESTEP), continueOnError=True)
                plt.figure()
                if "moreno" in self.NAME:  # Moreno plots summary statistics not current
                    plt.plot(self.DATA)
                    plt.plot(firstOut)
                    plt.plot(lastOut)
                    plt.ylabel('Summary Statistics')
                else:
                    plt.plot(np.arange(0, self.T_MAX, self.TIMESTEP), self.DATA)
                    plt.plot(np.arange(0, self.T_MAX, self.TIMESTEP), firstOut)
                    plt.plot(np.arange(0, self.T_MAX, self.TIMESTEP), lastOut)
                    plt.ylabel('Current')
                    plt.xlabel('Time (ms)')
                plt.legend(['Data', 'First Parameters', 'Best Parameters'])
                plt.title('Improvement after fitting: ' + self.NAME)
                plt.show()
            except IndexError:
                # If the tracker has no data (e.g. tracker.firstParams is empty), then we can't plot
                pass

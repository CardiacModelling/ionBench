import numpy as np
import myokit
import csv
import os
import ionbench
import matplotlib.pyplot as plt
import warnings
import pickle

class Tracker():
    """
    This class records the various performance metrics used to evaluate the optimisation algorithms. 
    
    It records the number of times the model is solved (stored as tracker.solveCount), not including the times that parameters were out of bounds.
    
    It records the RMSE (Root Mean Squared Error) cost each time a parameter vector is evaluated, using np.inf if parameters are out of bounds.
    
    It records the RMSRE (Root Mean Squared Relative Error) of the estimated parameters each time a parameter vector is evaluated. This is relative to the true parameters, ie the RMS of the vector of error between the estimated and true parameters, expressed as a percentage of the true parameters.
    
    It records the number of parameters that were correctly identified (within 5% of the true values) each time a parameter vector is evaluated.
    
    This class contains two methods: update(), and plot(). 
    
    update() is called everytime a parameter vector is simulated in the benchmarker (for example, in bm.cost) and updates the performance metric vectors.
    
    plot() is called during the benchmarkers .evaluate() method. It plots the performance metrics as functions of time (in the order in which parameter vectors were evaluated).
    """
    def __init__(self, trueParams):
        self.costs = []
        self.paramRMSRE = []
        self.paramIdentifiedCount = []
        self.solveCount = 0
        self.firstParams = []
        self.modelSolves = []
        self._trueParams = trueParams
    
    def update(self, estimatedParams, cost = np.inf, incrementSolveCounter = True):
        """
        This method updates the performance metric tracking vectors with new values. It should only need to be called by a benchmarker class.
        
        Parameters
        ----------
        
        estimatedParams : arraylike
            The vector of parameters that are being evaluated, after any transformations to return them to the original parameter space have been applied.
        cost : float, optional
            The RMSE cost of the parameter vectors that are being evaluated. The default is np.inf, to be used if parameters are out of bounds.
        incrementSolveCounter : bool, optional
            Should the solveCount be incremented, ie did the model need to be solved. This should only be False during benchmarker.evaluate() or if the parameters were out of bounds. The default is True.

        Returns
        -------
        None.

        """
        if len(self.firstParams)==0:
            self.firstParams = np.copy(estimatedParams)
        #Cast to numpy arrays
        trueParams = np.array(self._trueParams)
        estimatedParams = np.array(estimatedParams)
        #Update performance metrics
        self.paramRMSRE.append(np.sqrt(np.mean(((estimatedParams-trueParams)/trueParams)**2)))
        self.paramIdentifiedCount.append(np.sum(np.abs((estimatedParams-trueParams)/trueParams)<0.05))
        self.costs.append(cost)
        if incrementSolveCounter:
            self.solveCount += 1
        self.modelSolves.append(self.solveCount)
    
    def plot(self):
        """
        This method plots the performance metrics as functions of time (in the order in which parameter vectors were evaluated). It will produce three plots, the RMSE cost, parameter RMSRE, and the number of identified parameters over the optimisation. This method will be called when benchmarker.evaluate() is called, so long as benchmarker.plotter = True (the default).

        Returns
        -------
        None.

        """
        #Cost plot
        plt.figure()
        plt.scatter(range(len(self.costs)),self.costs, c="k", marker=".")
        plt.xlabel('Cost function calls')
        plt.ylabel('RMSE cost')
        plt.title('Data error')
        
        #Parameter RMSRE plot
        plt.figure()
        plt.scatter(range(len(self.paramRMSRE)),self.paramRMSRE, c="k", marker=".")
        plt.xlabel('Cost function calls')
        plt.ylabel('Parameter RMSRE')
        plt.title('Parameter error')
        
        #Number of identified parameters plot
        plt.figure()
        plt.scatter(range(len(self.paramIdentifiedCount)),self.paramIdentifiedCount, c="k", marker=".")
        plt.xlabel('Cost function calls')
        plt.ylabel('Number of parameters identified')
        plt.title('Number of parameters identified')
    
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
        data = (self.solveCount, self.costs, self.modelSolves, self.paramRMSRE, self.paramIdentifiedCount)
        with open(filename, 'wb') as f:
            pickle.dump(data,f)
    
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
        self.solveCount, self.costs, self.modelSolves, self.paramRMSRE, self.paramIdentifiedCount = data
    
    def report_convergence(self):
        """
        Reports the performance metrics at the point of convergence, defined as the first point where there is an unbroken chain in the number of correctly identified parameters.

        Returns
        -------
        None.

        """
        finalParamId = self.paramIdentifiedCount[-1]
        ifEqualFinalParamId = self.paramIdentifiedCount == finalParamId
        ind = [i for i, x in enumerate(ifEqualFinalParamId) if x] #Indexes where number of parameters identified is equal to the final count
        for i in ind:
            if all(ifEqualFinalParamId[i:]):
                #All future points remain with this many parameters identified, therefore it is considered converged
                print('Model solves until convergence:  '+str(self.modelSolves[i]))
                print('Cost at convergence:             {0:.6f}'.format(self.costs[i]))
                print('Parameter RMSRE at convergence:  {0:.6f}'.format(self.paramRMSRE[i]))
                break
    
class Benchmarker():
    """
    The Benchmarker class contains all the features needed to evaluate an optimisation algorithm. This class should not need to be called directly and is instead used as a parent class for the benchmarker problems. 
    
    The main methods to use from this class are n_parameters(), cost(), reset(), and evaluate().
    """
    def __init__(self):
        self._useScaleFactors = False
        self._bounded = False #Should the parameters be bounded
        self._logTransformParams = [False]*self.n_parameters() #Are any of the parameter log-transformed
        self.plotter = True #Should the performance metrics be plotted when evaluate() is called
        self.tracker = Tracker(self._trueParams) #Tracks the performance metrics
        
    def load_data(self, dataPath = '', paramPath = ''):
        """
        Loads output data to use in fitting.

        Parameters
        ----------
        dataPath : string, optional
            An absolute filepath to the .csv data file. The default is '', in which case no file will be loaded.
        paramPath : string, optional
            An absolute filepath to the .csv file containing the true parameters. The default is '', in which case no file will be loaded.

        Returns
        -------
        None.

        """
        if not dataPath == '':
            tmp=[]
            with open(dataPath, newline='') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    tmp.append(float(row[0]))
            self.data = np.array(tmp)
        if not paramPath == '':
            tmp=[]
            with open(paramPath, newline='') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    tmp.append(float(row[0]))
            self._trueParams = np.array(tmp)
    
    def add_bounds(self, bounds, parameterSpace = 'original'):
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
        self._bounded = True
    
    def log_transform(self, whichParams = []):
        """
        Fit some parameters in a log-transformed space. 
        
        Inputted log-transformed parameters will be set to exp(inputted parameters) before solving the model.

        Parameters
        ----------
        whichParams : list, optional
            Which parameters should be log-transformed, in the form of a list of bools, the same length as the number of parameters, where True is a parameter to be log-transformed. The default is [], in which case all parameters will be log-transformed.

        Returns
        -------
        None.

        """
        if whichParams == []: #Log-transform all parameters
            whichParams = [True]*self.n_parameters()
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
                parameters[i] = parameters[i]/self.defaultParams[i]
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
                parameters[i] = parameters[i]*self.defaultParams[i]
        
        return parameters
    
    def in_bounds(self, parameters):
        """
        Checks if parameters are inside any bounds. If benchmarker._bounded = False, then it always returns True.

        Parameters
        ----------
        parameters : list or numpy array
            Vector of parameters (in original parameter space) to check against the bounds.

        Returns
        -------
        bool
            True if parameters are inside the bounds or no bounds are specified, False if the parameters are outside of the bounds.

        """
        if self._bounded:
            if any(parameters[i]<self.lb[i] or parameters[i]>self.ub[i] for i in range(self.n_parameters())):
                return False
        return True
    
    def n_parameters(self):
        """
        Returns the number of parameters in the model.
        """
        return len(self.defaultParams)
    
    def reset(self):
        """
        Resets the benchmarker. This clears the simulation object and restarts the performance tracker.

        Returns
        -------
        None.

        """
        self.sim.reset()
        self.tracker = Tracker(self._trueParams)
    
    def cost(self, parameters, incrementSolveCounter = True):
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
        testOutput = np.array(self.simulate(parameters, np.arange(0, self.tmax), incrementSolveCounter = incrementSolveCounter))
        cost = np.sqrt(np.mean((testOutput-self.data)**2))
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
        #Calculate cost for a given set of parameters
        testOutput = np.array(self.simulate(parameters, np.arange(0, self.tmax)))
        return (testOutput-self.data)
    
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
        #Calculate cost for a given set of parameters
        testOutput = np.array(self.simulate(parameters, np.arange(0, self.tmax)))
        return (testOutput-self.data)**2
    
    def grad(self, parameters, centreCost = None, incrementSolveCounter = True):
        """
        Find the gradient of the RMSE cost at the inputted parameters through finite difference. If the problem is bounded, it will attempt to only evaluate parameters inside the bounds.

        Parameters
        ----------
        parameters : list or numpy array
            Vector of parameters to solve the model.
        centreCost : int, optional
            The cost evaluated at parameters. If this has already been calculated, it can be passed in as an input to avoid recalculating (reducing the solveCount). Default is None, in which case it will be automatically calculated.
        incrementSolveCounter : bool, optional
            If False, it disables the solve counter tracker. This never needs to be set to False by a user. This is only required by the evaluate() method. The default is True.

        Returns
        -------
        grad : array
            The gradient of the RMSE cost, evalauted around the inputted parameters.
        
        """
        def take_step(parameter, stepSize):
            # Takes a step in parameter space. If the parameter is 0, the step size is given by stepSize, otherwise the step size is stepSize*parameter. Returns the perturbed parameter
            if parameter == 0:
                return stepSize
            else:
                return parameter*(1+stepSize)
        #If inputted centre parameter vector is out of bounds, turn off bounds to avoid infinite costs
        boundsTurnedOff = False
        if not self.in_bounds(parameters):
            warnings.warn("Parameters at the centre of the gradient calculator were out of bounds. Bounds will be turned off to avoid an infinite cost being used in the finite difference calculation.")
            self._bounded = False
            boundsTurnedOff = True
        #If no centreCost was inputted, then calculate it
        if centreCost == None:
            centreCost = self.cost(parameters, incrementSolveCounter=incrementSolveCounter)
        #Find step sizes that ensure the parameters are always inside the bounds
        perturbationVector = np.zeros(self.n_parameters())
        for i in range(self.n_parameters()):
            stepSize = 1e-3 #Initial step size
            perturbedPoint = np.copy(parameters)
            perturbedPoint[i] = take_step(parameters[i],stepSize)
            #If the current step size results in out of bounds parameters, reduce and flip the step direction
            while not self.in_bounds(perturbedPoint):
                if np.abs(stepSize) < 1e-6:
                    #Avoid infinite loops
                    warnings.warn("Failed to find a perturbed parameter vector that respects the bounds. Bounds will be turned off to avoid an infinite cost being used in the finite difference calculation.")
                    self._bounded = False
                    boundsTurnedOff = True
                    break
                stepSize = stepSize/(-2) #Flip side and shrink step size
                perturbedPoint[i] = take_step(parameters[i],stepSize)
            #Save the working step size
            perturbationVector[i] = stepSize
        #Calculate gradient by finite differences
        grad = np.zeros(self.n_parameters())
        for i in range(self.n_parameters()):
            perturbedPoint = np.copy(parameters)
            perturbedPoint[i] = take_step(parameters[i],perturbationVector[i])
            perturbedCost = self.cost(perturbedPoint, incrementSolveCounter=incrementSolveCounter)
            #Calculate by finite difference
            grad[i] = (perturbedCost - centreCost)/(take_step(parameters[i],perturbationVector[i])-parameters[i])
        #If the bounds had to be turned off to calculate the gradient, turn them back on now that it has been calculated
        if boundsTurnedOff:
            self._bounded = True
        return grad
    
    def set_params(self, parameters):
        """
        Set the parameters in the simulation object. Inputted parameters should be in the original parameter space.

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
            self.sim.set_constant(self._paramContainer+'.p'+str(i+1), parameters[i])
    
    def solve_model(self, times, continueOnError = True):
        """
        Solve the model at the inputted times and return the current trace.

        Parameters
        ----------
        times : list or numpy array
            Vector of timepoints to record model output. Typically in ms.
        continueOnError : bool, optional
            If continueOnError is True, any errors that occur during solving the model will be ignored and an infinite output will be given. The default is True.

        Returns
        -------
        modelOutput : list
            A vector of model outputs (current trace).
        
        """
        if continueOnError:
            try:
                log = self.sim.run(self.tmax+1, log_times = times)
                return np.array(log[self._outputName])
            except:
                warnings.warn("Failed to solve model. Will report infinite output in the hope of continuing the run.")
                return np.array([np.inf]*len(times))
        else:
            log = self.sim.run(self.tmax+1, log_times = times)
            return np.array(log[self._outputName])
    
    def simulate(self, parameters, times, continueOnError = True, incrementSolveCounter = True):
        """
        Simulate the model for the inputted parameters and return the model output at the specified times.

        Parameters
        ----------
        parameters : list or numpy array
            Vector of parameters to solve the model.
        times : list or numpy array
            Vector of timepoints to record model output. Typically in ms.
        continueOnError : bool, optional
            If continueOnError is True, any errors that occur during solving the model will be ignored and an infinite output will be given. The default is True.
        incrementSolveCounter : bool, optional
            If False, it disables the solve counter tracker. This never needs to be set to False by a user. This is only required by the evaluate() method. The default is True.

        Returns
        -------
        modelOutput : list
            A vector of model outputs (current trace).

        """
        #Return the parameters to the original parameter space
        parameters = self.original_parameter_space(parameters) #Creates a copy of the parameter vector
        
        # Reset the simulation
        self.sim.reset()
        
        # Abort solving if the parameters are out of bounds
        if not self.in_bounds(parameters):
            self.tracker.update(parameters, incrementSolveCounter = False)
            if 'moreno' in self._name:
                return [np.inf]*69
            else:
                return [np.inf]*len(times)
        
        # Set the parameters in the simulation object
        self.set_params(parameters)
        
        # Run the simulation and track the performance
        out = self.solve_model(times, continueOnError = continueOnError)
        self.tracker.update(parameters, cost = np.sqrt(np.mean((out-self.data)**2)), incrementSolveCounter = incrementSolveCounter)
        return out
    
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
        print('Number of cost evaluations:      '+str(self.tracker.solveCount))
        cost =  self.cost(parameters, incrementSolveCounter = False)
        print('Final cost:                      {0:.6f}'.format(cost))
        print('Parameter RMSRE:                 {0:.6f}'.format(self.tracker.paramRMSRE[-1]))
        print('Number of identified parameters: '+str(self.tracker.paramIdentifiedCount[-1]))
        print('Total number of parameters:      '+str(self.n_parameters()))
        self.tracker.report_convergence()
        print('')
        if self.plotter:
            self.tracker.plot()
            self.sim.reset()
            self.set_params(self.tracker.firstParams)
            firstOut = self.solve_model(np.arange(self.tmax), continueOnError = True)
            self.sim.reset()
            self.set_params(self.original_parameter_space(parameters))
            lastOut = self.solve_model(np.arange(self.tmax), continueOnError = True)
            plt.figure()
            if "moreno" in self._name:
                plt.plot(self.data)
                plt.plot(firstOut)
                plt.plot(lastOut)
                plt.ylabel('Summary Statistics')
            else:
                plt.plot(np.arange(self.tmax),self.data)
                plt.plot(np.arange(self.tmax),firstOut)
                plt.plot(np.arange(self.tmax),lastOut)
                plt.ylabel('Current')
                plt.xlabel('Time (ms)')
            plt.legend(['Data','First Parameters','Final Parameters'])
            plt.title('Improvement after fitting: '+self._name)
            plt.show()

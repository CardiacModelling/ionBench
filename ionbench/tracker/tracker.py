import pickle
import warnings

import numpy as np
from matplotlib import pyplot as plt


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

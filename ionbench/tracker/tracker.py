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
        self.bestCosts = []
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
        self.bestCosts.append(self.bestCost)

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
        c = np.array(self.costs)
        plt.ylim(np.min(c[c < 1e5]), np.max(c[c < 1e5]))
        plt.xlabel('Model solves')
        plt.ylabel('RMSE cost')
        plt.title('Evaluated Cost')

        # Best cost plot
        plt.figure()
        plt.scatter(range(len(self.bestCosts)), self.bestCosts, c="k", marker=".")
        c = np.array(self.bestCosts)
        plt.ylim(np.min(c[c < 1e5]), np.max(c[c < 1e5]))
        plt.xlabel('Model solves')
        plt.ylabel('RMSE cost')
        plt.title('Best Cost')

        # # Parameter RMSRE plot
        # plt.figure()
        # plt.scatter(range(len(self.paramRMSRE)), self.paramRMSRE, c="k", marker=".")
        # plt.xlabel('Model solves')
        # plt.ylabel('Parameter RMSRE')
        # plt.title('Parameter error')
        #
        # # Number of identified parameters plot
        # plt.figure()
        # plt.scatter(range(len(self.paramIdentifiedCount)), self.paramIdentifiedCount, c="k", marker=".")
        # plt.xlabel('Model solves')
        # plt.ylabel('Number of parameters identified')
        # plt.title('Number of parameters identified')

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

        # Plot grad times
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
        data = {'solveCount': self.solveCount, 'gradSolveCount': self.gradSolveCount, 'costs': self.costs,
                'modelSolves': self.modelSolves, 'gradSolves': self.gradSolves, 'paramRMSE': self.paramRMSRE,
                'paramIdentifiedCount': self.paramIdentifiedCount, 'firstParams': self.firstParams, 'evals': self.evals,
                'bestParams': self.bestParams, 'bestCost': self.bestCost, 'bestCosts': self.bestCosts,
                'costTimes': self.costTimes,
                'gradTimes': self.gradTimes}
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
        keys = ['solveCount', 'gradSolveCount', 'costs', 'modelSolves', 'gradSolves', 'paramRMSE',
                'paramIdentifiedCount', 'firstParams', 'evals', 'bestParams', 'bestCost', 'bestCosts', 'costTimes',
                'gradTimes']
        try:
            self.solveCount, self.gradSolveCount, self.costs, self.modelSolves, self.gradSolves, self.paramRMSRE, self.paramIdentifiedCount, self.firstParams, self.evals, self.bestParams, self.bestCost, self.bestCosts, self.costTimes, self.gradTimes = [
                data[key] if key in data.keys() else None for key in keys]
        except AttributeError:
            # Assume old (v0.3.4) format
            self.solveCount, self.gradSolveCount, self.costs, self.modelSolves, self.gradSolves, self.paramRMSRE, self.paramIdentifiedCount, self.firstParams, self.evals, self.bestParams, self.bestCost, self.costTimes, self.gradTimes = data
            self.bestCosts = np.minimum.accumulate(self.costs)

    def report_convergence(self, threshold):
        """
        Reports the performance metrics at the point of convergence, defined using the two termination criteria: cost threshold and cost unchanged.

        Parameters
        ----------
        threshold : float
            The threshold to check if the cost was below.

        Returns
        -------
        None.

        """
        i = self.when_converged(threshold)
        if i is None:
            print('Optimiser has not converged.')
            print('Convergence reason:              Optimiser terminated early.')
        else:
            print('Convergence reason:              ' + 'Cost threshold' if self.cost_threshold(threshold,
                                                                                                i) else 'Cost unchanged')
            print('Cost evaluations at convergence: ' + str(self.modelSolves[i]))
            print('Grad evaluations at convergence: ' + str(self.gradSolves[i]))
            print('Cost at convergence:             {0:.6f}'.format(self.costs[i]))
            print('Parameter RMSRE at convergence:  {0:.6f}'.format(self.paramRMSRE[i]))

    def when_converged(self, threshold):
        """
        Returns the tracking index at convergence, defined using the two termination criteria: cost threshold and cost unchanged. For example, tracker.bestCosts[tracker.when_converged(threshold)] will return the best cost at convergence.

        Parameters
        ----------
        threshold : float
            The threshold to check if the cost was below in the cost_threshold termination criteria.

        Returns
        -------
        int
            The number of model solves at convergence. Returns None if the optimisation has not converged.
        """
        # Find cost unchanged using bisection search
        converged, unchangedIndex = self.cost_unchanged(returnIndex=True)
        unchangedIndex = unchangedIndex if converged else np.inf
        # Find cost threshold
        if self.cost_threshold(threshold):
            # First index where cost is below threshold
            thresholdIndex = np.argmax(np.array(self.bestCosts) < threshold)
        else:
            thresholdIndex = np.inf
        # Return the first index where one of the conditions is satisfied
        convergedIndex = min(unchangedIndex, thresholdIndex)
        if convergedIndex == np.inf:
            return None
        return convergedIndex

    def cost_threshold(self, threshold, index=None):
        """
        Checks if the cost threshold was satisfied at the given index. If no index is given, the last index is used.

        Parameters
        ----------
        threshold : float
            The threshold to check if the cost was below.
        index : int, optional
            The index to check if the cost threshold was satisfied. The default is None, in which case it checks the most recent parameter vector.
        Returns
        -------
        check : bool
            True if the function threshold was satisfied, False otherwise.
        """
        if index is None:
            index = len(self.bestCosts) - 1
        if index < 0:
            return False
        return self.bestCosts[index] < threshold

    def cost_unchanged(self, index=None, max_unchanged_evals=2500, returnIndex=False):
        """
        Checks if the cost had converged (remained unchanged for XX iterations) by the given index. If no index is given, the last index is used.

        Parameters
        ----------
        index : int, optional
            The index to check if the function threshold was satisfied. The default is None, in which case it checks the most recent parameter vector.
        max_unchanged_evals : int, optional
            The number of evaluations that the cost must remain unchanged for before it is considered converged. The default is 2500.
        returnIndex : bool, optional
            If True, returns the index at which the cost first became unchanged. The default is False. If convergence is not achieved, returns (False, None).

        Returns
        -------
        check : bool
            True if the function threshold was satisfied, False otherwise.
        """
        if index is None:
            index = len(self.bestCosts)
        if index < max_unchanged_evals:
            return False, None if returnIndex else False
        fsig = np.inf
        evalsUnchanged = 0
        for i in range(index):
            if np.abs(self.bestCosts[i] - fsig) > 1e-11:
                evalsUnchanged = 0
                fsig = self.bestCosts[i]
            else:
                evalsUnchanged += 1
            if evalsUnchanged >= max_unchanged_evals:
                return True, i if returnIndex else True
        return False, None if returnIndex else False

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

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

    def __init__(self):
        # Performance metrics over time
        self.costs = []  # Cost at evaluation (including from grad calls)
        self.costSolves = []  # Number of cost solves at a given evaluation
        self.gradSolves = []  # Number of grad solves at a given evaluation
        self.evals = []  # (evaluated parameters, solve type) at each evaluation
        self.bestCosts = []  # Best cost so far at each evaluation
        self.costTimes = []  # Time taken for cost solves at each evaluation (0 for grad solves)
        self.gradTimes = []  # Time taken for grad solves at each evaluation (0 for cost solves)

        # Best and first parameters
        self.bestParams = []  # Best parameters seen so far (lowest cost)
        self.bestCost = np.inf  # Best cost seen so far (includes cost from grad evaluations)
        self.firstParams = []  # First parameters, evaluated at the start of the optimisation

        # Counters
        self.costSolveCount = 0  # Current number of cost evaluations
        self.gradSolveCount = 0  # Current number of grad evaluations

        # Flags
        self.maxIterFlag = False  # Flag to indicate optimisation was cut short by maxIter or maxfev

    def update(self, estimatedParams, cost=np.inf, incrementSolveCounter=True, solveType='cost', solveTime=np.nan):
        """
        This method updates the performance metric tracking vectors with new values. It should only need to be called by a benchmarker class. Updates are not applied if the model did not need to be solved (because those parameters have been solved previously).

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
        # Cast to numpy array
        estimatedParams = np.array(estimatedParams)

        # Set first parameters if not already set
        if len(self.firstParams) == 0:
            self.firstParams = np.copy(estimatedParams)
            self.firstParams.flags['WRITEABLE'] = False

        # Only update lists if parameters actually needed to be solved (equivalent to caching)
        if not self.check_repeated_param(estimatedParams, solveType):
            # Update performance metrics
            self.costs.append(cost)
            # If the model was solved (anything but penalty function), increment the solve counter
            if incrementSolveCounter:
                if solveType == 'cost':
                    self.costSolveCount += 1
                elif solveType == 'grad':
                    self.gradSolveCount += 1
            else:
                # If not solved (penalty function), set solveTime to 0 and label as unsolved with solveType='none'
                solveTime = 0
                solveType = 'none'
            # Update list of times (update with 0 if different solve type)
            self.costTimes.append(solveTime if solveType == 'cost' else 0)
            self.gradTimes.append(solveTime if solveType == 'grad' else 0)

            # Update the list of evaluated parameters
            self.evals.append((estimatedParams, solveType))

            # Update the list of cost and grad solve counts
            self.costSolves.append(self.costSolveCount)
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

        def plot_times(times, name):
            """
            Plot the histograms of solve times.
            Parameters
            ----------
            times : list
                List of times to plot.
            name : string
                Name for title. Either 'cost' or 'gradient'
            """
            # Only attempt to plot if there is actually time data to plot
            if len(times) > 0:
                plt.figure()
                # Histogram of times
                n, _, _ = plt.hist(times)
                # Vertical line at mean
                plt.vlines(x=np.mean(times), ymin=0, ymax=np.max(n), colors=['k'],
                           label=f'Mean: {np.mean(times):.3f}')
                plt.xlabel('Time (sec)')
                plt.ylabel('Frequency')
                plt.title(f'Histogram of {name} evaluation times')
                plt.legend()
                plt.show()

        def plot_costs(costs, title):
            """
            Plot the costs.
            Parameters
            ----------
            costs : list
                List of costs to plot.
            title : string
                Title for the plot.
            """
            plt.figure()
            # Scatter plot of costs
            plt.scatter(range(len(costs)), costs, c="k", marker=".")
            c = np.array(costs)
            # Set y-axis limits to exclude out of bounds point if possible
            try:
                plt.ylim(np.min(c[c < 1e5]), np.max(c[c < 1e5]))
            except ValueError:  # pragma: no cover
                # All points out of bounds
                pass
            plt.xlabel('Model solves')
            plt.ylabel('RMSE cost')
            plt.title(title)
            plt.show()

        # Cost plot
        plot_costs(self.costs, 'Evaluated Cost')
        plot_costs(self.bestCosts, 'Best Costs')

        # Plot cost and grad times
        plot_times(self.costTimes, 'cost')
        plot_times(self.gradTimes, 'gradient')

    def save(self, filename):
        """
        Saves the tracked variables. Useful to store results to plot later.

        Parameters
        ----------
        filename : string
            Filename for storing the tracked variables. Variables will be pickled and stored in working directory.

        Returns
        -------
        None.

        """
        # Store data in a dictionary
        data = {'costSolveCount': self.costSolveCount, 'gradSolveCount': self.gradSolveCount, 'costs': self.costs,
                'costSolves': self.costSolves, 'gradSolves': self.gradSolves, 'firstParams': self.firstParams,
                'evals': self.evals, 'bestParams': self.bestParams, 'bestCost': self.bestCost,
                'bestCosts': self.bestCosts, 'costTimes': self.costTimes, 'gradTimes': self.gradTimes, 'maxIterFlag': self.maxIterFlag}
        # Pickle dictionary to save
        with open(filename, 'wb') as f:
            pickle.dump(data, f)

    def load(self, filename):
        """
        Loads the tracked variables. Useful to store results to plot later.

        Parameters
        ----------
        filename : string
            Filename to load the stored tracked variables. Variables will be read for the working directory. Must be saved using the .save() method associated with a tracker.

        Returns
        -------
        None.

        """
        # Load pickled dictionary
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        # Read off data from dictionary
        keys = ['costSolveCount', 'gradSolveCount', 'costs', 'costSolves', 'gradSolves', 'firstParams', 'evals', 'bestParams', 'bestCost', 'bestCosts', 'costTimes',
                'gradTimes', 'maxIterFlag']
        self.costSolveCount, self.gradSolveCount, self.costs, self.costSolves, self.gradSolves, self.firstParams, self.evals, self.bestParams, self.bestCost, self.bestCosts, self.costTimes, self.gradTimes, self.maxIterFlag = [
            data[key] if key in data.keys() else None for key in keys]

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
        # Find out when we converged to the cost threshold (None if didn't converge)
        i = self.when_converged(threshold)
        if i is None:
            # Check if we converged due to maxIter
            if self.maxIterFlag:
                print('Convergence reason:              Maximum iterations reached.')
            else:
                print('Convergence reason:              Unknown optimiser specific termination.')
            i = len(self.bestCosts) - 1
        else:
            print('Convergence reason:              ' + ('Cost threshold' if
                                                         self.cost_threshold(threshold, i) else 'Cost unchanged'))
        # Avoid printing if no data
        print('Cost evaluations at convergence: ' + str(self.costSolves[i] if len(self.costSolves) > 0 else None))
        print('Grad evaluations at convergence: ' + str(self.gradSolves[i] if len(self.gradSolves) > 0 else None))
        print('Best cost at convergence:        {0:.6f}'.format(self.bestCosts[i] if len(self.bestCosts) > 0 else self.bestCost))
        # Calculate total solve time at convergence
        if len(self.costSolves) > 0:
            costTime, gradTime = self.total_solve_time(i)
        else:
            costTime = 0
            gradTime = 0
        print('Model solve time at convergence: {0:.6f}'.format(costTime))
        print('Grad solve time at convergence:  {0:.6f}'.format(gradTime))

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
        # Check if cost has converged and find index if it has
        converged, unchangedIndex = self.cost_unchanged(returnIndex=True)
        unchangedIndex = unchangedIndex if converged else np.inf
        # Check if cost has reached threshold and find index if it has
        if self.cost_threshold(threshold):
            # First index where cost is below threshold
            thresholdIndex = np.argmax(np.array(self.bestCosts) < threshold)
        else:
            thresholdIndex = np.inf
        # Return the first index where one of the conditions is satisfied
        convergedIndex = min(unchangedIndex, thresholdIndex)
        # Return None if neither converged
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
        Checks if the cost had converged (remained unchanged for max_unchanged_evals function evaluations) by the given index. If no index is given, the last index is used.

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
        convergedIndex : int
            The index at which the cost first became unchanged. Returns None if did not converge. Only returned if returnIndex is True.
        """
        if index is None:
            index = len(self.bestCosts)
        # If we haven't reached max_unchanged_evals, then we can't have converged
        if index < max_unchanged_evals:
            return (False, None) if returnIndex else False
        fsig = np.inf
        evalsUnchanged = 0
        # Go through each index until it hasn't improved for max_unchanged_evals
        for i in range(index):
            if np.abs(self.bestCosts[i] - fsig) > 1e-7:
                evalsUnchanged = 0
                fsig = self.bestCosts[i]
            elif self.evals[i][-1] != 'none':
                # Only increment if the model is solved. Not if a penalty function is used.
                evalsUnchanged += 1
            if evalsUnchanged >= max_unchanged_evals:
                return (True, i) if returnIndex else True
        return (False, None) if returnIndex else False

    def total_solve_time(self, i):
        """
        Returns the total time taken to solve the model up to the i-th solve, separated into model solves with and without sensitivities.

        Parameters
        ----------
        i : int
            The solve index up to which the solve times should be totaled (inclusive).

        Returns
        -------
        costTime : float
            The total time taken to solve the model (excluding solves with sensitivities) up to (and including) the i-th solve.
        gradTime : float
            The total time taken to solve the model (excluding solves without sensitivities) up to (and including) the i-th solve.
        """
        if i is None:
            return np.sum(self.costTimes), np.sum(self.gradTimes)
        return np.sum(self.costTimes[:i + 1]), np.sum(self.gradTimes[:i + 1])

    def check_repeated_param(self, param, solveType):
        """
        Reports whether a parameter vector has been evaluated before in such a way that the current parameters did not need to be solved.

        Parameters
        ----------
        param : array
            The parameters to check if the model has been already solved for.
        solveType : string
            Specifies the type of solve that param corresponds to. 'grad' refers to solving with sensitivities and 'cost' refers to one without. The allows check_repeated_params to ignore any cases where the parameters are first solved as a cost and then again as a grad which is common, expected and optimal for line search based gradient descent methods.

        Returns
        -------
        repeat : bool
            True if the parameter vector has been evaluated before, False otherwise.

        """
        # Only care about repeated parameters if these parameters were solved
        if solveType != 'none':
            # Check against all previous parameters
            for (p, st) in self.evals:
                # Are the parameters equal and were they solved
                if all(p == param) and st != 'none':
                    if st == solveType:
                        # Previously evaluated the same parameter vector with the same solve type
                        return True
                    elif st == 'grad' and solveType == 'cost':
                        # Previously evaluated a grad which can automatically return the cost for free
                        return True
        return False

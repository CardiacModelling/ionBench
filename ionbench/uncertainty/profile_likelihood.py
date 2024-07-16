"""
Generates and plots profile likelihoods of benchmarker problems. Contains the class ProfileManager which masks a benchmarker but has a reduced parameter size. Contains the functions run(), which generates and saves the profile likelihoods, and plot_profile_likelihood(), which plots the saved results.
"""
import ionbench
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
import os
import scipy


class ProfileManager:
    """
    A middle-man for handling profile likelihood plots. This class acts as a wrapper for the benchmarker, such that the benchmarker always sees the full parameter vector, while the optimiser sees a reduced vector with one parameter missing.
    Contains methods for n_parameters(), cost(), grad(), signed_error(), set_params(), sample(), and evaluate(), all of which handle the mapping between the full set of parameters and the reduced optimised set of parameters.
    """

    def __init__(self, bm, fixedParam, fixedValue, x0):
        """
        Construct a ProfileManager.

        Parameters
        ----------
        bm : benchmarker
            A benchmarker to act as the wrapper for.
        fixedParam : int
            The index of the parameter to be fixed.
        fixedValue : float
            The value at which to fix the parameter.
        x0 : list
            Initial parameter vector (including the fixed parameter) to begin optimisation

        Returns
        -------
        None.

        """
        self.bm = bm
        self.fixedParam = fixedParam
        self.fixedValue = fixedValue
        self.MLE = x0
        self.parametersBounded = False
        self.NAME = bm.NAME
        self.T_MAX = bm.T_MAX
        self.TIMESTEP = bm.TIMESTEP
        self.DATA = bm.DATA

    def n_parameters(self):  # pragma: no cover
        """
        ProfileManager wrapper for the benchmarker's .n_parameters() method. Reduces the parameter count by 1 to account for the fixed parameter.
        """
        return self.bm.n_parameters() - 1

    def cost(self, params):  # pragma: no cover
        """
        ProfileManager wrapper for the benchmarker's .cost() method. Inserts the fixed parameter before evaluating.
        """
        return self.bm.cost(self.set_params(params))

    def grad(self, params, **kwargs):  # pragma: no cover
        """
        ProfileManager wrapper for the benchmarker's .grad() method. 
        """
        return np.delete(self.bm.grad(self.set_params(params), **kwargs), self.fixedParam, 1)

    def signed_error(self, params):  # pragma: no cover
        """
        ProfileManager wrapper for the benchmarker's .signed_error() method. Inserts the fixed parameter before evaluating.
        """
        return self.bm.signed_error(self.set_params(params))

    def set_params(self, params):  # pragma: no cover
        """
        Inserts the fixed parameter into the inputted parameter vector.
        """
        return np.insert(params, self.fixedParam, self.fixedValue)

    def sample(self):  # pragma: no cover
        """
        ProfileManager wrapper for the benchmarker's .sample() method. Since sampling is used to define the initial points, this method returns the true parameters with the fixed parameter removed.
        """
        return np.delete(self.MLE, self.fixedParam)

    def evaluate(self):  # pragma: no cover
        """
        ProfileManager wrapper for the benchmarker's .evaluate() method. This method does nothing, but is necessary since optimisers will attempt to call it.
        """
        pass


# noinspection PyUnboundLocalVariable,PyProtectedMember
def run(bm, variations, backwardPass=False, optimiser=ionbench.optimisers.scipy_optimisers.trustRegionReflective_scipy.run, filename=''):
    """
    Generate a profile likelihood style plot, reporting the fitted cost as a function of each fixed parameter.

    Parameters
    ----------
    bm : benchmarker
        A benchmarker function to explore.
    variations : list
        A list of parameter variations to apply for each parameter. It should be a list of length bm.n_parameters(), where the ith element is a list of variations to use to fix parameter i and fit the remaining parameters.
    backwardPass : bool, optional
        If False, the profile likelihood curve will be found by going left to right, using the optimised parameters from the left (lower variation) to initiate the optimisation. If True, it will travel right to left and the final name will be appended with 'B'. Both can be combined when plotting which can be advantageous to ensure a smooth profile likelihood plot. The default is False.
    optimiser: function, optional
        The optimiser to use for the profile likelihood plots. The default is trust region reflective.
    filename : string, optional
        A name to use to save the cost data. It will pickle the cost and variations and save them under [filename]_param[i].pickle. The default is '', in which case no data will be saved.

    Returns
    -------
    None.

    """
    bm.plotter = False
    if backwardPass:
        filename += 'B'
    for i in range(bm.n_parameters()):
        print('Working on parameter ' + str(i))
        costs = np.zeros(len(variations[i]))
        for j in range(len(variations[i])):
            bm.reset()
            if backwardPass:
                varIndex = len(variations[i]) - j - 1
            else:
                varIndex = j
            var = variations[i][varIndex]
            if j == 0:
                pm = ProfileManager(bm, i, bm._TRUE_PARAMETERS[i] * var, bm._TRUE_PARAMETERS)
            else:
                pm = ProfileManager(bm, i, bm._TRUE_PARAMETERS[i] * var, out)
            try:
                out = optimiser(pm)
                pm.MLE = bm._TRUE_PARAMETERS
                if pm.cost(out) >= 0.99 * pm.cost(
                        pm.sample()):  # If optimised cost not significantly better than unoptimised from default rates
                    pm.MLE = bm._TRUE_PARAMETERS
                    out = optimiser(pm)
            except Exception as e:  # pragma: no cover
                print(
                    'The optimisation caused an error (detailed below). In an attempt to recover, the profile likelihood will jump to the best guess.')
                print(e)
                out = pm.sample()

            costs[varIndex] = pm.cost(out)
            out = pm.set_params(out)
            print('Variation: ' + str(var))
            print('Cost found: ' + str(costs[varIndex]))
        if not filename == '':
            data = (variations[i], costs)
            with open(filename + '_param' + str(i) + '.pickle', 'wb') as f:
                pickle.dump(data, f)


# noinspection PyProtectedMember
def plot_profile_likelihood(modelType, numberToPlot, filepath='', debug=False):
    """
    Plot profile likelihood plots based on the pickled data in the current working directory.

    Parameters
    ----------
    modelType : string
        Type of model in benchmarker. This is used to load the benchmarker (options are hh, mm, ikr, ikur) and as the filename to load the pickled data [modelType]_param[i].pickle for i in range(numberToPlot)
    numberToPlot : int
        The number of profile likelihood plots to create. Should not exceed the number of parameters in the model.
    filepath : string, optional
        The filepath to a directory to save the figure inside. The default is '', in which case the figure will not be saved.
    debug : bool, optional
        If True, extra plots will be drawn to separate the forwards and backwards passes of the profile likelihood optimisation. The default is False.
    """
    # Initialise a benchmarker
    if modelType == 'hh':  # pragma: no cover
        bm = ionbench.problems.staircase.HH()
        title = 'Staircase HH'
    elif modelType == 'mm':  # pragma: no cover
        bm = ionbench.problems.staircase.MM()
        title = 'Staircase MM'
    elif modelType == 'ikr':  # pragma: no cover
        bm = ionbench.problems.loewe2016.IKr()
        title = 'Loewe IKr'
    elif modelType == 'ikur':  # pragma: no cover
        bm = ionbench.problems.loewe2016.IKur()
        title = 'Loewe IKur'
    elif modelType == 'ina':  # pragma: no cover
        bm = ionbench.problems.moreno2016.INa()
        title = 'Moreno INa'
    else:  # pragma: no cover
        bm = None
    # Use scale factors so only variations needs to be specified as the parameters
    bm.useScaleFactors = True

    perturbedCosts = []
    fig, axs = plt.subplots(int(np.ceil(numberToPlot/5)), 5, figsize=(7.5, np.ceil(numberToPlot/5)*1.5), layout='tight', sharey=True, sharex=True)
    minCost = np.inf
    maxCost = 0
    for i in range(numberToPlot):
        # Load the pickled data
        with open(modelType + '_param' + str(i) + '.pickle', 'rb') as f:
            variationsA, costsA = pickle.load(f)
            variations, costs = np.copy(variationsA), np.copy(costsA)
            # Central cost can be slightly off due to OS differences between data generation (Windows) and profile likelihood generation (Linux)
            if 'loewe' in bm.NAME or 'moreno' in bm.NAME:
                costs[variations == 1] = 0
        try:
            with open(modelType + 'B_param' + str(i) + '.pickle', 'rb') as f:
                variationsB, costsB = pickle.load(f)
            if len(variationsB) == len(variationsA):
                costs = np.array([min(costs[i], costsB[i]) for i in range(len(costs))])
        except FileNotFoundError:  # pragma: no cover
            pass

        if minCost > np.min(costs[costs > 0]):
            minCost = np.min(costs[costs > 0])
        if maxCost < np.max(costs[np.logical_and(costs < 1e5, costs > 0)]):
            maxCost = np.max(costs[np.logical_and(costs < 1e5, costs > 0)])

        # Calculate cost threshold
        lowCost = np.interp(0.95, variations, costs)
        if lowCost < 1e5:
            perturbedCosts.append(lowCost)
        highCost = np.interp(1.05, variations, costs)
        if highCost < 1e5:
            perturbedCosts.append(highCost)

        # Plot the profile likelihood
        axs[i//5, i % 5].semilogy(variations, costs, label='Optimised' if i == 0 else None, zorder=1)

        # If debug, plot forwards and backwards cost separately
        if debug:
            axs[i//5, i % 5].semilogy(variations, costsA, label='Forwards' if i == 0 else None, zorder=2, linestyle='dashed')
            try:
                if len(variationsB) == len(variationsA):
                    axs[i//5, i % 5].semilogy(variations, costsB, label='Backwards' if i == 0 else None, zorder=3, linestyle='dotted')
            except NameError:  # pragma: no cover
                pass
        axs[i//5, i % 5].set_xlabel(f'Parameter {i}')
        if i % 5 == 0:
            axs[i//5, i % 5].set_ylabel('Cost')

    # Get cost threshold
    threshold = scipy.stats.mstats.gmean(perturbedCosts)

    # Generate lots of tick options
    ylim = [10**(np.log10(minCost)-axs[0, 0].margins()[1]*(np.log10(maxCost)-np.log10(minCost))), 10**(np.log10(maxCost)+axs[0, 0].margins()[1]*(np.log10(maxCost)-np.log10(minCost)))]
    if maxCost/minCost < 1e3:
        # Small variation, finer scale is needed
        loc = mpl.ticker.LogLocator(numticks=20, subs=range(1, 10))
        possibleTicks = loc.tick_values(*ylim)

        # Find the first value in possibleTicks that is greater than ylim[0] and the last value that is less than ylim[1]
        yticks = []
        for i in range(len(possibleTicks)):
            if possibleTicks[i] > ylim[0]:
                yticks.append(possibleTicks[i])
                break
        for i in range(len(possibleTicks)-1, -1, -1):
            if possibleTicks[i] < ylim[1]:
                yticks.append(possibleTicks[i])
                break
        yticks = np.array(yticks)

    for i in range(numberToPlot):
        # Plot unoptimised cost slice
        costs = np.zeros(len(variations))
        for j in range(len(variations)):
            p = bm.input_parameter_space(bm._TRUE_PARAMETERS)
            p[i] = variations[j]
            costs[j] = bm.cost(p)
        axs[i//5, i % 5].semilogy(variations, costs, label='Unoptimised' if i == 0 else None, zorder=0)

    for i in range(numberToPlot, int(np.ceil(numberToPlot/5))*5):
        axs[i//5, i % 5].axis('off')

    if maxCost/minCost < 1e3:
        axs[0, 0].set_ylim(ylim)
        axs[0, 0].set_yticks(yticks)
    else:
        ylim = [10 ** np.floor(np.log10(ylim[0])), 10 ** np.ceil(np.log10(ylim[1]))]
        axs[0, 0].set_ylim(ylim)
        axs[0, 0].set_yticks(ylim)

    # Turn off minor ticks
    for ax in axs.flatten():
        ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())

    # add labels
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0))
    fig.suptitle(f'Profile likelihoods for {title}')
    # Save and show figure
    if filepath != '':
        fig.savefig(os.path.join(filepath, f'profileLikelihood-{modelType}'),
                    dpi=300, bbox_inches='tight')
    fig.show()

    # Print final cost threshold
    print(f'Threshold calculated as {threshold}')

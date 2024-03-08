"""
Generates and plots profile likelihoods of benchmarker problems. Contains the class ProfileManager which masks a benchmarker but has a reduced parameter size. Contains the functions run(), which generates and saves the profile likelihoods, and plot_profile_likelihood(), which plots the saved results.
"""
import ionbench
import numpy as np
import matplotlib.pyplot as plt
import pickle


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
        self._bounded = False

    def n_parameters(self):
        """
        ProfileManager wrapper for the benchmarker's .n_parameters() method. Reduces the parameter count by 1 to account for the fixed parameter.
        """
        return self.bm.n_parameters() - 1

    def cost(self, params):
        """
        ProfileManager wrapper for the benchmarker's .cost() method. Inserts the fixed parameter before evaluating.
        """
        return self.bm.cost(self.set_params(params))

    def grad(self, params, **kwargs):
        """
        ProfileManager wrapper for the benchmarker's .grad() method. 
        """
        return np.delete(self.bm.grad(self.set_params(params), **kwargs), self.fixedParam, 1)

    def signed_error(self, params):
        """
        ProfileManager wrapper for the benchmarker's .signed_error() method. Inserts the fixed parameter before evaluating.
        """
        return self.bm.signed_error(self.set_params(params))

    def set_params(self, params):
        """
        Inserts the fixed parameter into the inputted parameter vector.
        """
        return np.insert(params, self.fixedParam, self.fixedValue)

    def sample(self):
        """
        ProfileManager wrapper for the benchmarker's .sample() method. Since sampling is used to define the initial points, this method returns the true parameters with the fixed parameter removed.
        """
        return np.delete(self.MLE, self.fixedParam)

    def evaluate(self, x):
        """
        ProfileManager wrapper for the benchmarker's .evaluate() method. This method does nothing, but is necessary since optimisers will attempt to call it.
        """
        pass


# noinspection PyUnboundLocalVariable
def run(bm, variations, backwardPass=False, filename=''):
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
            if var == 1:
                costs[varIndex] = bm.cost(bm._TRUE_PARAMETERS)
                out = pm.set_params(pm.sample())
            else:
                try:
                    out = ionbench.optimisers.scipy_optimisers.lm_scipy.run(pm)
                    pm.MLE = bm._TRUE_PARAMETERS
                    if pm.cost(out) >= 0.99 * pm.cost(
                            pm.sample()):  # If optimised cost not significantly better than unoptimised from default rates
                        pm.MLE = bm._TRUE_PARAMETERS
                        out = ionbench.optimisers.scipy_optimisers.lm_scipy.run(pm)
                except Exception as e:
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


def plot_profile_likelihood(modelType, numberToPlot, debug=False):
    """
    Plot profile likelihood plots based on the pickled data in the current working directory.

    Parameters
    ----------
    modelType : string
        Type of model in benchmarker. This is used to load the benchmarker (options are hh, mm, ikr, ikur) and as the filename to load the pickled data [modelType]_param[i].pickle for i in range(numberToPlot)
    numberToPlot : int
        The number of profile likelihood plots to create. Should not exceed the number of parameters in the model.
    debug : bool, optional
        If True, extra plots will be drawn to separate the forwards and backwards passes of the profile likelihood optimisation. The default is False.
    """
    if modelType == 'hh':
        bm = ionbench.problems.staircase.HH()
    elif modelType == 'mm':
        bm = ionbench.problems.staircase.MM()
    elif modelType == 'ikr':
        bm = ionbench.problems.loewe2016.IKr()
    elif modelType == 'ikur':
        bm = ionbench.problems.loewe2016.IKur()
    else:
        bm = None
    bm._useScaleFactors = True
    ymin = np.inf
    ymax = 0
    for i in range(numberToPlot):
        with open(modelType + '_param' + str(i) + '.pickle', 'rb') as f:
            variations, costs = pickle.load(f)
        try:
            with open(modelType + 'B_param' + str(i) + '.pickle', 'rb') as f:
                variationsB, costsB = pickle.load(f)
            if len(variationsB) == len(variations):
                costs = np.array([min(costs[i], costsB[i]) for i in range(len(costs))])
        except FileNotFoundError:
            pass
        if np.max(costs[costs < np.inf]) > ymax:
            ymax = np.max(costs[costs < np.inf])
        if np.min(costs[costs > 1e-15]) < ymin:
            ymin = np.min(costs[costs > 1e-15])
    ymin /= 2
    ymax *= 5
    for i in range(numberToPlot):
        with open(modelType + '_param' + str(i) + '.pickle', 'rb') as f:
            variationsA, costsA = pickle.load(f)
            variations, costs = variationsA, costsA
        try:
            with open(modelType + 'B_param' + str(i) + '.pickle', 'rb') as f:
                variationsB, costsB = pickle.load(f)
            if len(variationsB) == len(variationsA):
                costs = np.array([min(costs[i], costsB[i]) for i in range(len(costs))])
        except FileNotFoundError:
            pass
        plt.figure()
        plt.semilogy(variations, costs, label='Optimised', zorder=1)
        costs = np.zeros(len(variations))
        for j in range(len(variations)):
            p = bm.input_parameter_space(bm._TRUE_PARAMETERS)
            p[i] = variations[j]
            costs[j] = bm.cost(p)
        plt.semilogy(variations, costs, label='Unoptimised', zorder=0)
        if debug:
            plt.semilogy(variations, costsA, label='Forwards', zorder=2, linestyle='dashed')
            try:
                if len(variationsB) == len(variationsA):
                    plt.semilogy(variations, costsB, label='Backwards', zorder=3, linestyle='dotted')
            except NameError:
                pass
        plt.ylim(ymin, ymax)
        plt.title('Profile likelihood: ' + modelType)
        plt.xlabel('Factor for parameter ' + str(i))
        plt.ylabel('Cost')
        plt.legend()
        plt.show()

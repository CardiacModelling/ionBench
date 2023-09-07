import ionbench
import numpy as np
import matplotlib.pyplot as plt
import pickle

class ProfileManager():
    def __init__(self, bm, fixedParam, fixedValue):
        """
        A middle-man for handling profile likelihood plots. This class acts as a wrapper for the benchmarker, such that the benchmarker always sees the full parameter vector, while the optimiser sees a reduced vector with one parameter missing.

        Parameters
        ----------
        bm : benchmarker
            A benchmarker to act as the wrapper for.
        fixedParam : int
            The index of the parameter to be fixed.
        fixedValue : float
            The value at which to fix the parameter.

        Returns
        -------
        None.

        """
        self.bm = bm
        self.fixedParam = fixedParam
        self.fixedValue = fixedValue
        self.MLE = bm._trueParams
        self._bounded = False
    
    def n_parameters(self):
        """
        ProfileManager wrapper for the benchmarker's .n_parameters() method. Reduces the parameter count by 1 to account for the fixed parameter.
        """
        return self.bm.n_parameters()-1
    
    def cost(self, params):
        """
        ProfileManager wrapper for the benchmarker's .cost() method. Inserts the fixed parameter before evaluating.
        """
        return self.bm.cost(self.set_params(params))
    
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
        ProfileManager wrapper for the benchmarker's .evaluate() method. This method does nothing, but is neccessary since optimisers will attempt to call it.
        """
        pass

def profile_likelihood(bm, variations, plot = True, filename = ''):
    """
    Generate a profile likelihood style plot, reporting the fitted cost as a function of each fixed parameter.

    Parameters
    ----------
    bm : benchmarker
        A benchmarker function to explore.
    variations : list
        A list of parameter variations to apply for each parameter. It should be a list of length bm.n_parameters(), where the ith element is a list of variations to use to fix parameter i and fit the remaining parameters. 
    plot : bool, optional
        Whether or not to plot the likelihood and give plots of the current at the start and end points to see if the variability is significant. The default is True.
    filename : string, optional
        A name to use to save the cost data. It will pickle the cost and variations and save them under [filename]_param[i].pickle. The default is '', in which case no data will be saved.

    Returns
    -------
    None.

    """
    bm.plotter = False
    for i in range(bm.n_parameters()):
        print('Working on parameter '+str(i))
        costs = np.zeros(len(variations[i]))
        for j in range(len(variations[i])):
            bm.reset()
            var = variations[i][j]
            pm = ProfileManager(bm, i, bm._trueParams[i]*var, bm._trueParams)
            if var == 1:
                costs[j] = bm.cost(bm._trueParams)
                out = pm.sample()
            else:
                try:
                    out = ionbench.optimisers.scipy_optimisers.lm_scipy.run(pm)
                except:
                    out = pm.sample()
                costs[j] = pm.cost(out)
            print('Variation: '+str(var))
            print('Cost found: '+str(costs[j]))
            if j==0 or j==len(variations[i])-1 and plot:
                bm.sim.reset()
                bm.set_params(pm.set_params(out))
                curr1 = bm.solve_model(np.arange(bm.tmax), continueOnError = True)
                bm.sim.reset()
                bm.set_params(bm._trueParams)
                curr2 = bm.solve_model(np.arange(bm.tmax), continueOnError = True)
                plt.figure()
                plt.plot(np.arange(bm.tmax),curr1,'k')
                plt.plot(np.arange(bm.tmax),curr2,'b')
                plt.ylabel('Current')
                plt.xlabel('Time (ms)')
                plt.legend(['Fitted','Data'])
                plt.show()
        if plot:
            plt.figure()
            plt.plot(variations[i], costs)
            plt.xlabel('Parameter '+str(i))
            plt.title('Profile Likelihood')
            plt.ylabel('Cost')
            plt.show()
        if not filename == '':
            data = (variations, costs)
            with open(filename+'_param'+str(i)+'.pickle', 'wb') as f:
                pickle.dump(data,f)

def run_all():
    """
    Generate all of the profile likelihoods and save them to files to plot later.

    Returns
    -------
    None.

    """
    var = lambda x:np.linspace(1-x,1+x,2)
    variations = [var(0.2)]*4 + [var(0.4)]*4 + [var(0.2)] #aVar = 0.2, bVar = 0.4
    bm = ionbench.problems.staircase.HH_Benchmarker()
    profile_likelihood(bm, variations, filename = 'hh')
    
    #MM doesn't seem to change. Might be unidentifiable - worth checking eigenvalues to see how many parameters can be identified
    variations = [var(0.8), var(0.8), var(0.8), var(0.8), var(0.8), var(0.8), var(0.8), var(0.8), var(0.8), var(0.8), var(0.8), var(0.8), var(0.8), var(0.8)]
    bm = ionbench.problems.staircase.MM_Benchmarker()
    profile_likelihood(bm, variations, filename = 'mm')
    
    variations = [var(0.99), var(2), var(1.5), var(2), var(1.5), var(1.2), var(0.99), var(0.99), var(0.99), var(0.6), var(0.7), var(0.7)]
    bm = ionbench.problems.loewe2016.ikr()
    profile_likelihood(bm, variations, filename = 'ikr')
    
    variations = [var(0.99), var(2), var(0.99), var(2), var(0.99), var(2), var(2), var(0.99), var(2), var(0.99), var(0.99), var(0.99), var(2), var(2), var(0.99), var(2), var(2), var(2), var(0.99), var(0.99), var(2), var(0.99), var(2), var(0.99), var(0.99)]
    bm = ionbench.problems.loewe2016.ikur()
    profile_likelihood(bm, variations, filename = 'ikur')

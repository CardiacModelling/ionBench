import os
import numpy as np
import csv
import myokit
import ionBench

class Staircase_Benchmarker(ionBench.benchmarker.Benchmarker):
    def __init__(self):
        self._log = myokit.DataLog.load_csv(os.path.join(ionBench.DATA_DIR, 'staircase', 'staircase-ramp.csv'))
        self._useScaleFactors = True
        super().__init__()
        try:
            self.loadData(os.path.join(ionBench.DATA_DIR, 'staircase', 'data'+self._modelType+'.csv'), os.path.join(ionBench.DATA_DIR, 'staircase', 'trueParams'+self._modelType+'.csv'))
        except FileNotFoundError:
            self.data = None
            self._trueParams = None
    
    def sample(self, n=1):
        """
        Sample parameters for the staircase problems.

        Parameters
        ----------
        n : int, optional
            Number of parameter vectors to sample. The default is 1.

        Returns
        -------
        params : list
            If n=1, then params is the vector of parameters. Otherwise, params is a list containing n parameter vectors.

        """
        params = [None]*n
        for i in range(n):
            params[i] = np.random.uniform(0.5,1.5,self.n_parameters())
        if n==1:
            return params[0]
        else:
            return params
        
class HH_Benchmarker(Staircase_Benchmarker):
    """
    The Hodgkin-Huxley IKr Staircase benchmarker. 
    
    The benchmarker uses the Beattie et al 2017 IKr Hodgkin-Huxley model using the staircase protocol. 
    
    Its parameters are specified as scaling factors, so start at a vector of all ones, or sample from the benchmarker.sample() method. 
    """
    def __init__(self):
        print('Initialising Hodgkin-Huxley IKr benchmark')
        self.model = myokit.load_model(os.path.join(ionBench.DATA_DIR, 'staircase', 'beattie-2017-ikr-hh.mmt'))
        self._outputName = 'ikr.IKr'
        self._paramContainer = 'ikr'
        self._modelType = 'HH'
        self.defaultParams = [2.26e-4, 0.0699, 3.45e-5, 0.05462, 0.0873, 8.91e-3, 5.15e-3, 0.03158, 0.1524]
        super().__init__()
        print('Benchmarker initialised')

class MM_Benchmarker(Staircase_Benchmarker):
    """
    The Markov IKr Staircase benchmarker. 
    
    The benchmarker uses the Fink et al 2008 IKr Markov model using the staircase protocol. 
    
    Its parameters are specified as scaling factors, so start at a vector of all ones, or sample from the benchmarker.sample() method. 
    """
    def __init__(self):
        print('Initialising Markov Model IKr benchmark')
        self.model = myokit.load_model(os.path.join(ionBench.DATA_DIR, 'staircase', 'fink-2008-ikr-mm.mmt'))
        self._outputName = 'IKr.i_Kr'
        self._paramContainer = 'iKr_Markov'
        self._modelType = 'MM'
        self.defaultParams = [0.20618, 0.0112, 0.04209, 0.02202, 0.0365, 0.41811, 0.0223, 0.13279, -0.0603, 0.08094, 0.0002262, -0.0399, 0.04150, -0.0312]
        super().__init__()
        print('Benchmarker initialised')

def generateData(modelType):
    """
    Generate the data files for the staircase benchmarker problems.

    Parameters
    ----------
    modelType : string
        'HH' to generate the data for the Hodgkin-Huxley benchmark problem. 'MM' to generate the data for the Markov model benchmark problem.

    Returns
    -------
    None.

    """
    modelType = modelType.upper()
    if modelType == 'HH':
        bm = HH_Benchmarker()
    elif modelType == 'MM':
        bm = MM_Benchmarker()
    trueParams = np.random.uniform(0.5,1.5,bm.n_parameters())
    out = bm.simulate(trueParams, np.arange(bm.tmax), continueOnError = False)
    out = out + np.random.normal(0, np.mean(np.abs(out))*0.05, len(out))
    with open(os.path.join(ionBench.DATA_DIR, 'staircase', 'data'+modelType+'.csv'), 'w', newline = '') as csvfile:
        writer = csv.writer(csvfile, delimiter = ',')
        writer.writerows(map(lambda x: [x], out))
    with open(os.path.join(ionBench.DATA_DIR, 'staircase', 'trueParams'+modelType+'.csv'), 'w', newline = '') as csvfile:
        writer = csv.writer(csvfile, delimiter = ',')
        writer.writerows(map(lambda x: [x], trueParams))
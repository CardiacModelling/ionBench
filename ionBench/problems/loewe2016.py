import ionBench
import myokit
import os
import numpy as np
import csv

class loewe2016_Benchmarker(ionBench.benchmarker.Benchmarker):
    def __init__(self):
        self._log = myokit.DataLog.load_csv(os.path.join(ionBench.DATA_DIR, 'loewe2016', 'protocol.csv'))
        self._trueParams = self.defaultParams
        self.paramSpaceWidth = 1 #1 for narrow, 2 for wide
        self._useScaleFactors = False
        super().__init__()
    
    def sample(self, n=1):
        """
        Sample parameters for the Loewe 2016 problems. By default the sampling using the narrow parameter space but this can be changed by setting benchmarker.paramSpaceWidth = 2 to use the wide parameter space.

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
            param = [None]*self.n_parameters()
            for j in range(self.n_parameters()):
                if self.additiveParams[j]:
                    param[j] = self.defaultParams[j] + np.random.uniform(-60*self.paramSpaceWidth,60*self.paramSpaceWidth)
                else:
                    param[j] = self.defaultParams[j]*10**np.random.uniform(-1*self.paramSpaceWidth,1*self.paramSpaceWidth) #Log uniform distribution
            params[i] = param
        if n==1:
            return params[0]
        else:
            return params

class ikr(loewe2016_Benchmarker):
    """
    The Loewe 2016 IKr benchmarker. 
    
    The benchmarker uses the Courtemanche 1998 IKr model with a simple step protocol. 
    
    Its parameters are specified as reported in Loewe et al 2016 with the true parameters being the same as the default and the center of the sampling distribution. 
    """
    def __init__(self):
        print('Initialising Loewe 2016 IKr benchmark')
        self.model = myokit.load_model(os.path.join(ionBench.DATA_DIR, 'loewe2016', 'courtemanche-1998-ikr.mmt'))
        self._outputName = 'ikr.IKr'
        self._paramContainer = 'ikr'
        self.defaultParams = np.array([3e-4, 14.1, 5, 3.3328, 5.1237, 1, 14.1, 6.5, 15, 22.4, 0.029411765, 138.994])
        self.additiveParams = [False, True, False, True, False, False, True, False, True, False, False, False]
        self._rateFunctions = [(lambda p,V:p[0]*(V+p[1])/(1-np.exp((V+p[1])/(-p[2]))), 'positive'), (lambda p,V:7.3898e-5*(V+p[3])/(np.exp((V+p[3])/p[4])-1), 'negative')] #Used for rate bounds
        self.loadData(dataPath = os.path.join(ionBench.DATA_DIR, 'loewe2016', 'ikr.csv'))
        super().__init__()
        print('Benchmarker initialised')

class ikur(loewe2016_Benchmarker):
    """
    The Loewe 2016 IKur benchmarker. 
    
    The benchmarker uses the Courtemanche 1998 IKur model with a simple step protocol. 
    
    Its parameters are specified as reported in Loewe et al 2016 with the true parameters being the same as the default and the center of the sampling distribution. 
    """
    def __init__(self):
        print('Initialising Loewe 2016 IKur benchmark')
        self.model = myokit.load_model(os.path.join(ionBench.DATA_DIR, 'loewe2016', 'courtemanche-1998-ikur.mmt'))
        self._outputName = 'ikur.IKur'
        self._paramContainer = 'ikur'
        self.defaultParams = np.array([0.65, 10, 8.5, 30, 59, 2.5, 82, 17, 30.3, 9.6, 3, 1, 21, 185, 28, 158, 16, 99.45, 27.48, 3, 0.005, 0.05, 15, 13, 138.994])
        self.additiveParams = [False, True, False, True, False, True, True, False, True, False, False, False, True, True, False, True, True, True, False, False, True, False, True, False, False]
        self._rateFunctions = [(lambda p,V: p[0]/(np.exp((V+p[1])/-p[2])+np.exp((V-p[3])/-p[4])), 'positive'), (lambda p,V: 0.65/(p[5]+np.exp((V+p[6])/p[7])), 'negative'), (lambda p,V: p[11]/(p[12]+np.exp((V-p[13])/-p[14])), 'positive'), (lambda p,V: np.exp((V-p[15])/-p[16]), 'negative')] #Used for rate bounds
        self.loadData(dataPath = os.path.join(ionBench.DATA_DIR, 'loewe2016', 'ikur.csv'))
        super().__init__()
        self.sim.set_tolerance(1e-12,1e-12)
        print('Benchmarker initialised')

def generateData(modelType):
    """
    Generate the data files for the Loewe 2016 benchmarker problems. The true parameters are the same as the deafult for these benchmark problems.

    Parameters
    ----------
    modelType : string
        'ikr' to generate the data for the IKr benchmark problem. 'ikur' to generate the data for the IKur benchmark problem.

    Returns
    -------
    None.

    """
    modelType = modelType.lower()
    if modelType == 'ikr':
        bm = ikr()
    elif modelType == 'ikur':
        bm = ikur()
    out = bm.simulate(bm.defaultParams, np.arange(bm.tmax), continueOnError = False)
    with open(os.path.join(ionBench.DATA_DIR, 'loewe2016', modelType.lower()+'.csv'), 'w', newline = '') as csvfile:
        writer = csv.writer(csvfile, delimiter = ',')
        writer.writerows(map(lambda x: [x], out))


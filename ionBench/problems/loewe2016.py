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
        super().__init__()
    
    def sample(self, n=1):
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
    
    def setParams(self, parameters):
        # Update the parameters
        for i in range(self.n_parameters()):
            self.sim.set_constant(self._paramContainer+'.p'+str(i+1), parameters[i])

class ikr(loewe2016_Benchmarker):
    def __init__(self):
        print('Initialising Loewe 2016 IKr benchmark')
        self.model = myokit.load_model(os.path.join(ionBench.DATA_DIR, 'loewe2016', 'courtemanche-1998-ikr.mmt'))
        self._outputName = 'ikr.IKr'
        self._paramContainer = 'ikr'
        self.defaultParams = np.array([3e-4, 14.1, 5, 3.3328, 5.1237, 1, 14.1, 6.5, 15, 22.4, 0.029411765, 138.994])
        self.additiveParams = [False, True, False, True, False, False, True, False, True, False, False, False]
        self.loadData(dataPath = os.path.join(ionBench.DATA_DIR, 'loewe2016', 'ikr.csv'))
        super().__init__()
        print('Benchmarker initialised')

class ikur(loewe2016_Benchmarker):
    def __init__(self):
        print('Initialising Loewe 2016 IKur benchmark')
        self.model = myokit.load_model(os.path.join(ionBench.DATA_DIR, 'loewe2016', 'courtemanche-1998-ikur.mmt'))
        self._outputName = 'ikur.IKur'
        self._paramContainer = 'ikur'
        self.defaultParams = np.array([0.65, 10, 8.5, 30, 59, 2.5, 82, 17, 30.3, 9.6, 3, 1, 21, 185, 28, 158, 16, 99.45, 27.48, 3, 0.005, 0.05, 15, 13, 138.994])
        self.additiveParams = [False, True, False, True, False, True, True, False, True, False, False, False, True, True, False, True, True, True, False, False, True, False, True, False, False]
        self.loadData(dataPath = os.path.join(ionBench.DATA_DIR, 'loewe2016', 'ikur.csv'))
        super().__init__()
        self.sim.set_tolerance(1e-12,1e-12)
        print('Benchmarker initialised')

def generateData(modelType):
    modelType = modelType.lower()
    if modelType == 'ikr':
        bm = ikr()
    elif modelType == 'ikur':
        bm = ikur()
    out = bm.simulate(bm.defaultParams, np.arange(bm.tmax), continueOnError = False)
    with open(os.path.join(ionBench.DATA_DIR, 'loewe2016', modelType.lower()+'.csv'), 'w', newline = '') as csvfile:
        writer = csv.writer(csvfile, delimiter = ',')
        writer.writerows(map(lambda x: [x], out))


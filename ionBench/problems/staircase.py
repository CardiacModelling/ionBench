import os
import numpy as np
import csv
import myokit
import ionBench

class HH_Benchmarker(ionBench.benchmarker.Benchmarker):
    def __init__(self):
        print('Initialising Hodgkin-Huxley IKr benchmark')
        self.model = myokit.load_model(os.path.join(ionBench.DATA_DIR, 'staircase', 'beattie-2017-ikr-hh.mmt'))
        self._log = myokit.DataLog.load_csv(os.path.join(ionBench.DATA_DIR, 'staircase', 'staircase-ramp.csv'))
        self._outputName = 'ikr.IKr'
        self._paramContainer = 'ikr'
        super().__init__()
        self.defaultParams = [2.26e-4, 0.0699, 3.45e-5, 0.05462, 0.0873, 8.91e-3, 5.15e-3, 0.03158, 0.1524]
        try:
            self.loadData(os.path.join(ionBench.DATA_DIR, 'staircase', 'dataHH.csv'), os.path.join(ionBench.DATA_DIR, 'staircase', 'trueParamsHH.csv'))
        except FileNotFoundError:
            self.data = None
            self._trueParams = None
        print('Benchmarker initialised')

class MM_Benchmarker(ionBench.benchmarker.Benchmarker):
    def __init__(self):
        print('Initialising Markov Model IKr benchmark')
        self.model = myokit.load_model(os.path.join(ionBench.DATA_DIR, 'staircase', 'fink-2008-ikr-mm.mmt'))
        self._log = myokit.DataLog.load_csv(os.path.join(ionBench.DATA_DIR, 'staircase', 'staircase-ramp.csv'))
        self._outputName = 'IKr.i_Kr'
        self._paramContainer = 'iKr_Markov'
        super().__init__()
        self.defaultParams = [0.20618, 0.0112, 0.04209, 0.02202, 0.0365, 0.41811, 0.0223, 0.13279, -0.0603, 0.08094, 0.0002262, -0.0399, 0.04150, -0.0312]
        try:
            self.loadData(os.path.join(ionBench.DATA_DIR, 'staircase', 'dataMM.csv'), os.path.join(ionBench.DATA_DIR, 'staircase', 'trueParamsMM.csv'))
        except FileNotFoundError:
            self.data = None
            self._trueParams = None
        print('Benchmarker initialised')

def generateData(modelType):
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
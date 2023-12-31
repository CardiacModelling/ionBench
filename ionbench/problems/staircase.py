import os
import numpy as np
import csv
import myokit
import myokit.lib.markov
import myokit.lib.hh
import ionbench


class Staircase_Benchmarker(ionbench.benchmarker.Benchmarker):
    def __init__(self):
        self._log = myokit.DataLog.load_csv(os.path.join(ionbench.DATA_DIR, 'staircase', 'staircase-ramp.csv'))
        self._trueParams = np.copy(self.defaultParams)
        try:
            self.load_data(os.path.join(ionbench.DATA_DIR, 'staircase', 'data' + self._modelType + '.csv'))
        except FileNotFoundError:
            self.data = None
        if self.sensitivityCalc:
            paramNames = [self._paramContainer + '.p' + str(i + 1) for i in range(self.n_parameters())]
            sens = ([self._outputName], paramNames)
            self.simSens = myokit.Simulation(self.model, sensitivities=sens, protocol=self.protocol())
            self.simSens.set_tolerance(1e-8, 1e-8)
        else:
            self.simSens = None
        self.sim = myokit.Simulation(self.model, protocol=self.protocol())
        self.sim.set_tolerance(1e-8, 1e-8)
        super().__init__()

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
        params = [None] * n
        for i in range(n):
            params[i] = self.input_parameter_space(self.defaultParams * np.random.uniform(0.5, 1.5, self.n_parameters()))
        if n == 1:
            return params[0]
        else:
            return params

    def protocol(self):
        protocol = myokit.TimeSeriesProtocol(self._log.time(), self._log['voltage'])
        self.tmax = self._log.time()[-1]
        return protocol


class HH_Benchmarker(Staircase_Benchmarker):
    """
    The Hodgkin-Huxley IKr Staircase benchmarker.

    The benchmarker uses the Beattie et al 2017 IKr Hodgkin-Huxley model using the staircase protocol.

    Its parameters are specified as scaling factors, so start at a vector of all ones, or sample from the benchmarker.sample() method.
    """

    def __init__(self, sensitivities=False):
        print('Initialising Hodgkin-Huxley IKr benchmark')
        self._name = "staircase.hh"
        self.model = myokit.load_model(os.path.join(ionbench.DATA_DIR, 'staircase', 'beattie-2017-ikr-hh.mmt'))
        self._outputName = 'ikr.IKr'
        self._paramContainer = 'ikr'
        self._modelType = 'HH'
        self.defaultParams = np.array([2.26e-4, 0.0699, 3.45e-5, 0.05462, 0.0873, 8.91e-3, 5.15e-3, 0.03158, 0.1524])
        self._rateFunctions = [(lambda p, V:p[0] * np.exp(p[1] * V), 'positive'), (lambda p, V:p[2] * np.exp(-p[3] * V), 'negative'), (lambda p, V:p[4] * np.exp(p[5] * V), 'positive'), (lambda p, V:p[6] * np.exp(-p[7] * V), 'negative')]  # Used for rate bounds
        self.standardLogTransform = [True, False] * 4 + [False]
        self.sensitivityCalc = sensitivities
        self._analyticalModel = myokit.lib.hh.HHModel(model=self.model, states=['ikr.act', 'ikr.rec'], parameters=[self._paramContainer + '.p' + str(i + 1) for i in range(self.n_parameters())], current=self._outputName, vm='membrane.V')
        super().__init__()
        print('Benchmarker initialised')


class MM_Benchmarker(Staircase_Benchmarker):
    """
    The Markov IKr Staircase benchmarker.

    The benchmarker uses the Fink et al 2008 IKr Markov model using the staircase protocol.

    Its parameters are specified as scaling factors, so start at a vector of all ones, or sample from the benchmarker.sample() method.
    """

    def __init__(self, sensitivities=False):
        print('Initialising Markov Model IKr benchmark')
        self._name = "staircase.mm"
        self.model = myokit.load_model(os.path.join(ionbench.DATA_DIR, 'staircase', 'fink-2008-ikr-mm.mmt'))
        self._outputName = 'IKr.i_Kr'
        self._paramContainer = 'iKr_Markov'
        self._modelType = 'MM'
        self.defaultParams = np.array([0.20618, 0.0112, 0.04209, 0.02202, 0.0365, 0.41811, 0.0223, 0.13279, 0.0603, 0.08094, 0.0002262, 0.0399, 0.04150, 0.0312])
        self._rateFunctions = [(lambda p, V:p[0] * np.exp(p[1] * V), 'positive'), (lambda p, V:p[2], 'independent'), (lambda p, V:p[3] * np.exp(p[4] * V), 'positive'), (lambda p, V:p[5] * np.exp(p[6] * V), 'positive'), (lambda p, V:p[7] * np.exp(-p[8] * V), 'negative'), (lambda p, V:p[9], 'independent'), (lambda p, V:p[10] * np.exp(-p[11] * V), 'negative'), (lambda p, V:p[12] * np.exp(-p[13] * V), 'negative')]  # Used for rate bounds
        self.standardLogTransform = [True, False, True] * 2 + [False, True] * 2 + [True, False] * 2
        self.sensitivityCalc = sensitivities
        self._analyticalModel = myokit.lib.markov.LinearModel(model=self.model, states=['iKr_Markov.' + s for s in ['Cr1', 'Cr2', 'Cr3', 'Or4', 'Ir5']], parameters=[self._paramContainer + '.p' + str(i + 1) for i in range(self.n_parameters())], current=self._outputName, vm='Environment.V')
        super().__init__()
        print('Benchmarker initialised')


def generate_data(modelType):
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
    out = bm.simulate(bm._trueParams, np.arange(bm.tmax), continueOnError=False)
    out += np.random.normal(0, np.mean(np.abs(out)) * 0.05, len(out))
    with open(os.path.join(ionbench.DATA_DIR, 'staircase', 'data' + modelType + '.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows(map(lambda x: [x], out))
    with open(os.path.join(ionbench.DATA_DIR, 'staircase', 'trueParams' + modelType + '.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows(map(lambda x: [x], bm._trueParams))

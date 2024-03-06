"""
Contains the two staircase benchmarker problems, HH and MM. These both inherit from StaircaseBenchmarker which itself inherits from benchmarker.Benchmarker.
generate_data(modelType) will generate the data for either the HH or MM problem and store it in the data directory.
"""
import os
import numpy as np
import csv
import myokit
import myokit.lib.markov
import myokit.lib.hh
import ionbench
import scipy.stats
import copy


class StaircaseBenchmarker(ionbench.benchmarker.Benchmarker):
    def __init__(self):
        self.tmax = None
        self.tols = (1e-11, 1e-10)
        try:
            self.load_data(os.path.join(ionbench.DATA_DIR, 'staircase', 'data' + self._modelType + '.csv'))
        except FileNotFoundError:
            self.data = None
        if self.sensitivityCalc:
            paramNames = [self._paramContainer + '.p' + str(i + 1) for i in range(self.n_parameters())]
            sens = ([self._outputName], paramNames)
            self.simSens = myokit.Simulation(self.model, sensitivities=sens, protocol=self.protocol())
            self.simSens.set_tolerance(self.tols[0], self.tols[1])
        else:
            self.simSens = None
        self.sim = myokit.Simulation(self.model, protocol=self.protocol())
        self.sim.set_tolerance(self.tols[0], self.tols[1])
        self.freq = 0.5  # Timestep in data between points
        self.rateMin = 1.67e-5
        self.rateMax = 1e3
        p = self.protocol()
        self.vLow = min(p.levels())
        self.vHigh = max(p.levels())
        self.lbStandard = np.array([1e-7] * (self.n_parameters() - 1) + [0.02])
        self.ubStandard = np.array(
            [1e3 if self.standardLogTransform[i] else 0.4 for i in range(self.n_parameters() - 1)] + [0.2])
        self.lb = copy.copy(self.lbStandard)
        self.ub = copy.copy(self.ubStandard)
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
            while True:
                p = []
                for j in range(self.n_parameters()):
                    if self.standardLogTransform[j]:
                        p.append(scipy.stats.loguniform.rvs(self.lbStandard[j], self.ubStandard[j]))
                    else:
                        p.append(np.random.uniform(self.lbStandard[j], self.ubStandard[j]))
                if self.in_rate_bounds(p, boundedCheck=False):
                    break
            params[i] = self.input_parameter_space(p)
        if n == 1:
            return params[0]
        else:
            return params

    def protocol(self):
        """
        Gets the staircase voltage protocol from the loaded log and returns it. Setting self.tmax to the length of the protocol.
        Returns
        -------
        protocol : myokit.Protocol
            The staircase protocol.
        """
        protocol = myokit.load_protocol(os.path.join(ionbench.DATA_DIR, 'staircase', 'staircase-pace.mmt'))
        self.tmax = protocol.characteristic_time()
        return protocol

    def add_ramps(self):
        """
        Myokit protocols do not support ramps, so this method is used to add the staircase ramps to the myokit model.
        """
        c = self.model.get('membrane')
        # Remove binding from membrane.V
        v = c.get('V')
        v.set_binding(None)

        # Ramps
        v1 = c.add_variable('v1')
        v1.set_rhs('-150 + 0.1 * engine.time')
        v2 = c.add_variable('v2')
        v2.set_rhs('5694 - 0.4 * engine.time')

        # membrane.V is pace with ramps
        v.set_rhs("""
            piecewise(
                (engine.time >= 300 and engine.time < 700), v1, 
                (engine.time >=14410 and engine.time < 14510), v2, 
                engine.pace)
        """)


class HH(StaircaseBenchmarker):
    """
    The Hodgkin-Huxley IKr Staircase benchmarker.

    The benchmarker uses the Beattie et al. 2017 IKr Hodgkin-Huxley model using the staircase protocol.

    Its parameters are specified as scaling factors, so start at a vector of all ones, or sample from the benchmarker.sample() method.
    """

    def __init__(self, sensitivities=False):
        print('Initialising Hodgkin-Huxley IKr benchmark')
        self._name = "staircase.hh"
        self.model = myokit.load_model(os.path.join(ionbench.DATA_DIR, 'staircase', 'beattie-2017-ikr-hh.mmt'))
        self.add_ramps()
        self._outputName = 'ikr.IKr'
        self._paramContainer = 'ikr'
        self._modelType = 'HH'
        self.defaultParams = np.array([2.26e-4, 0.0699, 3.45e-5, 0.05462, 0.0873, 8.91e-3, 5.15e-3, 0.03158, 0.1524])
        self._rateFunctions = [(lambda p, V: p[0] * np.exp(p[1] * V), 'positive'),
                               (lambda p, V: p[2] * np.exp(-p[3] * V), 'negative'),
                               (lambda p, V: p[4] * np.exp(p[5] * V), 'positive'),
                               (lambda p, V: p[6] * np.exp(-p[7] * V), 'negative')]  # Used for rate bounds
        self.standardLogTransform = [True, False] * 4 + [False]
        self.sensitivityCalc = sensitivities
        self._analyticalModel = myokit.lib.hh.HHModel(model=self.model, states=['ikr.act', 'ikr.rec'],
                                                      parameters=[self._paramContainer + '.p' + str(i + 1) for i in
                                                                  range(self.n_parameters())], current=self._outputName,
                                                      vm='membrane.V')
        self.costThreshold = 0.02
        super().__init__()
        print('Benchmarker initialised')


class MM(StaircaseBenchmarker):
    """
    The Markov IKr Staircase benchmarker.

    The benchmarker uses the Fink et al. 2008 IKr Markov model using the staircase protocol.

    Its parameters are specified as scaling factors, so start at a vector of all ones, or sample from the benchmarker.sample() method.
    """

    def __init__(self, sensitivities=False):
        print('Initialising Markov Model IKr benchmark')
        self._name = "staircase.mm"
        self.model = myokit.load_model(os.path.join(ionbench.DATA_DIR, 'staircase', 'fink-2008-ikr-mm.mmt'))
        self.add_ramps()
        self._outputName = 'IKr.i_Kr'
        self._paramContainer = 'iKr_Markov'
        self._modelType = 'MM'
        self.defaultParams = np.array(
            [0.20618, 0.0112, 0.04209, 0.02202, 0.0365, 0.41811, 0.0223, 0.13279, 0.0603, 0.08094, 0.0002262, 0.0399,
             0.04150, 0.0312, 0.024])
        self._rateFunctions = [(lambda p, v: p[0] * np.exp(p[1] * v), 'positive'),
                               (lambda p, v: p[2], 'independent'),
                               (lambda p, v: p[3] * np.exp(p[4] * v), 'positive'),
                               (lambda p, v: p[5] * np.exp(p[6] * v), 'positive'),
                               (lambda p, v: p[7] * np.exp(-p[8] * v), 'negative'),
                               (lambda p, v: p[9], 'independent'),
                               (lambda p, v: p[10] * np.exp(-p[11] * v), 'negative'),
                               (lambda p, v: p[12] * np.exp(-p[13] * v), 'negative')]  # Used for rate bounds
        self.standardLogTransform = [True, False, True] * 2 + [False, True] * 2 + [True, False] * 2 + [False]
        self.sensitivityCalc = sensitivities
        self._analyticalModel = myokit.lib.markov.LinearModel(model=self.model, states=['iKr_Markov.' + s for s in
                                                                                        ['Cr1', 'Cr2', 'Cr3', 'Or4',
                                                                                         'Ir5']],
                                                              parameters=[self._paramContainer + '.p' + str(i + 1) for i
                                                                          in range(self.n_parameters())],
                                                              current=self._outputName, vm='membrane.V')
        self.costThreshold = 0.0075
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
        bm = HH()
    else:
        bm = MM()
    bm.set_params(bm.defaultParams)
    bm.set_steady_state(bm.defaultParams)
    out = bm.solve_model(np.arange(0, bm.tmax, bm.freq), continueOnError=False)
    out += np.random.normal(0, np.mean(np.abs(out)) * 0.05, len(out))
    with open(os.path.join(ionbench.DATA_DIR, 'staircase', 'data' + modelType + '.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows(map(lambda x: [x], out))

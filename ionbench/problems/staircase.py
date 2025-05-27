"""
Contains the two staircase benchmarker problems, HH and MM. These both inherit from StaircaseBenchmarker which itself inherits from benchmarker.Benchmarker.
generate_data(modelType) will generate the data for either the HH or MM problem and store it in the data directory.
"""
import csv
import os

import myokit
import numpy as np
from scipy.stats import loguniform
from myokit.lib.hh import HHModel
from myokit.lib.markov import LinearModel

import ionbench


# noinspection PyUnresolvedReferences
class StaircaseBenchmarker(ionbench.benchmarker.Benchmarker):
    def __init__(self):
        # Benchmarker
        p = self.protocol()
        self.TIMESTEP = 0.5  # Timestep in data between points
        self.T_MAX = p.characteristic_time()
        self._LOWER_BOUND = np.array([1e-7] * (self.n_parameters() - 1) + [0.02])
        self._UPPER_BOUND = np.array(
            [1e3 if self.STANDARD_LOG_TRANSFORM[i] else 0.4 for i in range(self.n_parameters() - 1)] + [0.2])
        self.lb = np.copy(self._LOWER_BOUND)
        self.ub = np.copy(self._UPPER_BOUND)
        self.RATE_MIN = 1.67e-5
        self.RATE_MAX = 1e3
        try:
            self.load_data(
                os.path.join(ionbench.DATA_DIR, 'staircase', 'data' + ('HH' if 'hh' in self.NAME else 'MM') + '.csv'))
        except FileNotFoundError:  # pragma: no cover
            self.DATA = None

        # Myokit
        self.sim = myokit.Simulation(self._MODEL, protocol=p)
        self.sim.set_tolerance(*self._TOLERANCES)
        if self.sensitivityCalc:
            paramNames = [self._PARAMETER_CONTAINER + '.p' + str(i + 1) for i in range(self.n_parameters())]
            sens = ([self._OUTPUT_NAME], paramNames)
            self.simSens = myokit.Simulation(self._MODEL, sensitivities=sens, protocol=p)
            self.simSens.set_tolerance(*self._TOLERANCES)
        else:
            self.simSens = None
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
                    if self.STANDARD_LOG_TRANSFORM[j]:
                        p.append(loguniform.rvs(self._LOWER_BOUND[j], self._UPPER_BOUND[j]))
                    else:
                        p.append(np.random.uniform(self._LOWER_BOUND[j], self._UPPER_BOUND[j]))
                if self.in_rate_bounds(p, boundedCheck=False):
                    break
            params[i] = self.input_parameter_space(p)
        if n == 1:
            return params[0]
        else:
            return params

    @staticmethod
    def protocol():
        """
        Gets the staircase voltage protocol from the loaded log and returns it. Setting self.T_MAX to the length of the protocol.

        Returns
        -------
        protocol : myokit.Protocol
            The staircase protocol.
        """
        protocol = myokit.load_protocol(os.path.join(ionbench.DATA_DIR, 'staircase', 'staircase-pace.mmt'))
        return protocol

    @staticmethod
    def add_ramps(model):
        """
        Myokit protocols do not support ramps, so this method is used to add the staircase ramps to the myokit model. This does not copy the model and will modify the input.

        Parameters
        ----------
        model : myokit.Model
            The model to add the ramps to.

        Returns
        -------
        model : myokit.Model
            The model with the ramps added.
        """
        c = model.get('membrane')
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
        return model


class HH(StaircaseBenchmarker):
    """
    The Hodgkin-Huxley IKr Staircase benchmarker.

    The benchmarker uses the Beattie et al. 2017 IKr Hodgkin-Huxley model using the staircase protocol.

    Its parameters are specified as scaling factors, so start at a vector of all ones, or sample from the benchmarker.sample() method.
    """

    def __init__(self, sensitivities=False):
        print('Initialising Hodgkin-Huxley IKr benchmark')
        # Benchmarker
        self.NAME = "staircase.hh"
        self._TRUE_PARAMETERS = np.array([2.26e-4, 0.0699, 3.45e-5, 0.05462, 0.0873, 8.91e-3, 5.15e-3, 0.03158, 0.1524])
        self.STANDARD_LOG_TRANSFORM = (True, False) * 4 + (False,)
        self._RATE_FUNCTIONS = (lambda p, V: p[0] * np.exp(p[1] * V),
                                lambda p, V: p[2] * np.exp(-p[3] * V),
                                lambda p, V: p[4] * np.exp(p[5] * V),
                                lambda p, V: p[6] * np.exp(-p[7] * V))  # Used for rate bounds
        self.sensitivityCalc = sensitivities
        self.COST_THRESHOLD = 1.57e-2

        # Myokit
        model = myokit.load_model(os.path.join(ionbench.DATA_DIR, 'staircase', 'beattie-2017-ikr-hh.mmt'))
        self._MODEL = self.add_ramps(model)
        self._OUTPUT_NAME = 'ikr.IKr'
        self._PARAMETER_CONTAINER = 'ikr'
        self._ANALYTICAL_MODEL = HHModel(model=self._MODEL, states=['ikr.act', 'ikr.rec'], current=self._OUTPUT_NAME,
                                         parameters=[self._PARAMETER_CONTAINER + '.p' + str(i + 1) for i in
                                                     range(self.n_parameters())], vm='membrane.V')
        self._TOLERANCES = (1e-5, 1e-5)

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
        # Benchmarker
        self.NAME = "staircase.mm"
        self._TRUE_PARAMETERS = np.array(
            [0.20618, 0.0112, 0.04209, 0.02202, 0.0365, 0.41811, 0.0223, 0.13279, 0.0603, 0.08094, 0.0002262, 0.0399,
             0.04150, 0.0312, 0.024])
        self.STANDARD_LOG_TRANSFORM = (True, False, True) * 2 + (False, True) * 2 + (True, False) * 2 + (False,)
        self._RATE_FUNCTIONS = (lambda p, V: p[0] * np.exp(p[1] * V),
                                lambda p, V: p[2],
                                lambda p, V: p[3] * np.exp(p[4] * V),
                                lambda p, V: p[5] * np.exp(p[6] * V),
                                lambda p, V: p[7] * np.exp(-p[8] * V),
                                lambda p, V: p[9],
                                lambda p, V: p[10] * np.exp(-p[11] * V),
                                lambda p, V: p[12] * np.exp(-p[13] * V))  # Used for rate bounds
        self.sensitivityCalc = sensitivities
        self.COST_THRESHOLD = 5.77e-3

        # Myokit
        model = myokit.load_model(os.path.join(ionbench.DATA_DIR, 'staircase', 'fink-2008-ikr-mm.mmt'))
        self._MODEL = self.add_ramps(model)
        self._OUTPUT_NAME = 'IKr.i_Kr'
        self._PARAMETER_CONTAINER = 'iKr_Markov'
        self._ANALYTICAL_MODEL = LinearModel(model=self._MODEL, current=self._OUTPUT_NAME, vm='membrane.V',
                                             states=['iKr_Markov.' + s for s in ['Cr1', 'Cr2', 'Cr3', 'Or4', 'Ir5']],
                                             parameters=[self._PARAMETER_CONTAINER + '.p' + str(i + 1) for i in
                                                         range(self.n_parameters())])
        self._TOLERANCES = (1e-6, 1e-6)
        super().__init__()
        print('Benchmarker initialised')


# noinspection PyProtectedMember
def generate_data(modelType, noiseStrength=0.05):  # pragma: no cover
    """
    Generate the data files for the staircase benchmarker problems.

    Parameters
    ----------
    modelType : string
        'HH' to generate the data for the Hodgkin-Huxley benchmark problem. 'MM' to generate the data for the Markov model benchmark problem.
    noiseStrength : float, optional
        The strength of the noise to add to the data. The default is 0.05, which is 5% of the mean absolute value of the data.

    Returns
    -------
    None.

    """
    modelType = modelType.upper()
    if modelType == 'HH':
        bm = HH()
    else:
        bm = MM()
    bm.set_params(bm._TRUE_PARAMETERS)
    bm.set_steady_state(bm._TRUE_PARAMETERS)
    out = bm.solve_model(np.arange(0, bm.T_MAX, bm.TIMESTEP), continueOnError=False)
    out += np.random.normal(0, np.mean(np.abs(out)) * noiseStrength, len(out))
    with open(os.path.join(ionbench.DATA_DIR, 'staircase', 'data' + modelType + '.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows(map(lambda x: [x], out))

"""
Contains the two benchmarker problems from Loewe et al. 2016, IKr and IKur. These both inherit from LoeweBenchmarker which itself inherits from benchmarker.Benchmarker.
generate_data(modelType) will generate the data for either the IKr or IKur problem and store it in the data directory.
"""
import ionbench
import myokit
import myokit.lib.hh
import os
import numpy as np
import csv


class LoeweBenchmarker(ionbench.benchmarker.Benchmarker):
    def __init__(self, states):
        # Benchmarker
        p = self.protocol()
        self.TIMESTEP = 0.5  # Timestep in data between points
        self.T_MAX = p.characteristic_time()  # Length of protocol
        # Parameter bounds (hidden from user)
        self._LOWER_BOUND = np.array([self._TRUE_PARAMETERS[i] * 0.1 if self.STANDARD_LOG_TRANSFORM[i] else
                                      self._TRUE_PARAMETERS[i] - 60 for i in range(self.n_parameters())])
        self._UPPER_BOUND = np.array([self._TRUE_PARAMETERS[i] * 10 if self.STANDARD_LOG_TRANSFORM[i] else
                                      self._TRUE_PARAMETERS[i] + 60 for i in range(self.n_parameters())])
        # Parameter bounds (available to user)
        self.lb = np.copy(self._LOWER_BOUND)
        self.ub = np.copy(self._UPPER_BOUND)
        # Rate bounds
        self.RATE_MIN = 1.67e-5
        self.RATE_MAX = 1e3

        # Myokit
        parameters = [self._PARAMETER_CONTAINER + '.p' + str(i + 1) for i in range(self.n_parameters())]
        self._ANALYTICAL_MODEL = myokit.lib.hh.HHModel(model=self._MODEL, states=states, parameters=parameters,
                                                       current=self._OUTPUT_NAME, vm='membrane.V')
        self.sim = myokit.lib.hh.AnalyticalSimulation(self._ANALYTICAL_MODEL, protocol=p)
        if self.sensitivityCalc:
            # ODE solver
            sens = ([self._OUTPUT_NAME], parameters)
            self.simSens = myokit.Simulation(self._MODEL, sensitivities=sens, protocol=p)
            self.simSens.set_tolerance(*self._TOLERANCES)
        else:
            self.simSens = None
        super().__init__()

    def sample(self, n=1):
        """
        Sample parameters for the Loewe 2016 problems.

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
            param = [None] * self.n_parameters()
            for j in range(self.n_parameters()):
                if self.STANDARD_LOG_TRANSFORM[j]:
                    param[j] = self._TRUE_PARAMETERS[j] * 10 ** np.random.uniform(-1, 1)
                else:
                    param[j] = self._TRUE_PARAMETERS[j] + np.random.uniform(-60, 60)
            params[i] = self.input_parameter_space(param)  # Generates a copy
        if n == 1:
            return params[0]
        else:
            return params

    @staticmethod
    def protocol():
        """
        Add the protocol from Loewe et al. 2016. This protocol consists of 20ms at -80mV, a variable height step between 50mV and -70mV (in steps of 10mV) for 400ms and then 400ms at -110mV before repeating with a new height for the central step (in order of decreasing voltage).

        Returns
        -------
        p : myokit.Protocol
            A myokit.Protocol for the voltage clamp protocol from Loewe et al. 2016.

        """
        p = myokit.Protocol()
        vsteps = []
        for i in range(13):
            vsteps.append(-80)
            vsteps.append(50 - i * 10)
            vsteps.append(-110)
        durations = [20, 400, 400] * 13
        for i in range(len(vsteps)):
            p.add_step(vsteps[i], durations[i])
        return p


class IKr(LoeweBenchmarker):
    """
    The Loewe 2016 IKr benchmarker.

    The benchmarker uses the Courtemanche 1998 IKr model with a simple step protocol.

    Its parameters are specified as reported in Loewe et al. 2016 with the true parameters being the same as the default and the center of the sampling distribution.
    """

    def __init__(self, sensitivities=False):
        print('Initialising Loewe 2016 IKr benchmark')
        # Benchmarker
        self.NAME = "loewe2016.ikr"
        self._TRUE_PARAMETERS = np.array([3e-4, 14.1, 5, 3.3328, 5.1237, 1, 14.1, 6.5, 15, 22.4, 0.029411765, 138.994])
        self.STANDARD_LOG_TRANSFORM = (True, False, True, False, True, True, False, True, False, True, True, True)
        self._RATE_FUNCTIONS = (lambda p, V: p[0] * (V + p[1]) / (1 - np.exp((V + p[1]) / (-p[2]))),
                                lambda p, V: 7.3898e-5 * (V + p[3]) / (np.exp((V + p[3]) / p[4]) - 1))  # Used for rate bounds
        self.sensitivityCalc = sensitivities
        self.COST_THRESHOLD = 2.95e-6

        # Myokit
        self._MODEL = myokit.load_model(os.path.join(ionbench.DATA_DIR, 'loewe2016', 'courtemanche-1998-ikr.mmt'))
        self._OUTPUT_NAME = 'ikr.IKr'
        self._PARAMETER_CONTAINER = 'ikr'
        self._TOLERANCES = (1e-6, 1e-6)
        self.load_data(dataPath=os.path.join(ionbench.DATA_DIR, 'loewe2016', 'ikr.csv'))
        super().__init__(states=['ikr.xr'])
        print('Benchmarker initialised')


class IKur(LoeweBenchmarker):
    """
    The Loewe 2016 IKur benchmarker.

    The benchmarker uses the Courtemanche 1998 IKur model with a simple step protocol.

    Its parameters are specified as reported in Loewe et al. 2016 with the true parameters being the same as the default and the center of the sampling distribution.
    """

    def __init__(self, sensitivities=False):
        print('Initialising Loewe 2016 IKur benchmark')
        # Benchmarker
        self.NAME = "loewe2016.ikur"
        self._TRUE_PARAMETERS = np.array(
            [0.65, 10, 8.5, 30, 59, 2.5, 82, 17, 30.3, 9.6, 3, 1, 21, 185, 28, 158, 16, 99.45, 27.48, 3, 0.005, 0.05,
             15, 13, 138.994])
        self.STANDARD_LOG_TRANSFORM = (True, False, True, False, True, False, False, True, False, True, True, True,
                                       False, False, True, False, True, False, True, True, False, True, False, True,
                                       True)
        self._RATE_FUNCTIONS = (
            lambda p, V: p[0] / (np.exp((V + p[1]) / -p[2]) + np.exp((V - p[3]) / -p[4])),
            lambda p, V: 0.65 / (p[5] + np.exp((V + p[6]) / p[7])),
            lambda p, V: p[11] / (p[12] + np.exp((V - p[13]) / -p[14])),
            lambda p, V: np.exp((V - p[15]) / p[16]))  # Used for rate bounds
        self.sensitivityCalc = sensitivities
        self.COST_THRESHOLD = 2.80e-7

        # Myokit
        self._MODEL = myokit.load_model(os.path.join(ionbench.DATA_DIR, 'loewe2016', 'courtemanche-1998-ikur.mmt'))
        self._OUTPUT_NAME = 'ikur.IKur'
        self._PARAMETER_CONTAINER = 'ikur'
        self._TOLERANCES = (1e-8, 1e-8)
        self.load_data(dataPath=os.path.join(ionbench.DATA_DIR, 'loewe2016', 'ikur.csv'))
        super().__init__(states=['ikur.ua', 'ikur.ui'])
        print('Benchmarker initialised')


# noinspection PyProtectedMember
def generate_data(modelType):  # pragma: no cover
    """
    Generate the data files for the Loewe 2016 benchmarker problems. The true parameters are the same as the default for these benchmark problems.

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
        bm = IKr()
    else:
        bm = IKur()
    bm.set_params(bm._TRUE_PARAMETERS)
    bm.set_steady_state(bm._TRUE_PARAMETERS)
    out = bm.solve_model(np.arange(0, bm.T_MAX, bm.TIMESTEP), continueOnError=False)
    with open(os.path.join(ionbench.DATA_DIR, 'loewe2016', modelType + '.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows(map(lambda x: [x], out))

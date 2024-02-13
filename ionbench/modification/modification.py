"""
Contains the main Modification class and all the paper-specific modification subclasses.
"""
import ionbench
import numpy as np


# noinspection PyProtectedMember
class Modification:
    """
    Modifications provide a way to store some benchmarker settings and apply them to a range of benchmarkers. These settings include log-transforms, parameter bounds, and the use of scale factors. A single modification can be generated, and then applied to a range of benchmarkers. Once a Modification object is instantiated, its transforms and bounds can be applied to a benchmarker, bm, by calling modification.apply(bm)
    """

    def __init__(self, name="", logTransform='None', bounds='None', scaleFactors='off', customLogTransform=None,
                 customBounds=None, kwargs=None):
        """
        Initialise a Modification object. Once a Modification is built, its setting can be applied to any benchmarker, bm, by calling modification.apply(bm)

        Parameters
        ----------
        name : string, optional
            A name for the modification. Useful for logging. The default is "".
        logTransform : string, optional
            Setting for log transforms. Options are 'None' (default, none of the parameters will be log transformed), 'Standard' (a problem specific set of parameters will be log transformed), 'Full' (all parameters will be log transformed), and 'Custom' (Log transformed parameters specified by the customLogTransform input will be log transformed).
        bounds : string, optional
            Setting for parameter bounds. Options are 'None' (default, none of the parameters will be bounds), 'Positive' (all parameters will have a lower bound of 0 and no upper bound), 'Sampler' (problem specific bounds using the range in the benchmarkers sampler will be used), and 'Custom' (Bounds on parameters specified by the customBounds input will be used).
        scaleFactors : string, optional
            Setting for scale factors. Options are 'off' (default, scale factors won't be used), and 'on' (scale factors will be used, with the default parameters representing 1).
        customLogTransform : list, optional
            A list of parameters to log transform. The list should be the same length as the number of parameters in the benchmarker. This makes this option problem specific and will be unlikely to be usable across different benchmarkers. Only required if logTransform = 'Custom'.
        customBounds : list, optional
            A list containing 2 elements, a list of lower bounds and a list of upper bounds. Each sub-list should be the same length as the number of parameters in the benchmarker. This makes this option problem specific and will be unlikely to be usable across different benchmarkers. Only required if bounds = 'Custom'.
        kwargs : dict, optional
            A dictionary of keyword arguments to be passed into an optimisers run function. The default is None.

        Returns
        -------
        None.

        """
        if kwargs is None:
            kwargs = {}
        self.dict = {'log transform': logTransform, 'bounds': bounds, 'scale factors': scaleFactors}
        self.customLogTransform = customLogTransform
        self.customBounds = customBounds
        if len(name) > 0:
            self._name = name
        else:
            self._name = None
        self.kwargs = kwargs

    def apply(self, bm):
        """
        Applies the settings in this modification to the inputted benchmarker. This will override any previous settings assigned to the benchmarker.

        Parameters
        ----------
        bm : benchmarker
            A benchmarker problem to apply this modification.

        Returns
        -------
        None.

        """
        self.apply_log_transforms(self.dict['log transform'], bm)
        self.apply_bounds(self.dict['bounds'], bm)
        self.apply_scale_factors(self.dict['scale factors'], bm)

    def apply_log_transforms(self, setting, bm):
        """
        Apply the log transform setting to a benchmarker.

        Parameters
        ----------
        setting : string
            Setting for log transforms. Options are 'None' (default, none of the parameters will be log transformed), 'Standard' (a problem specific set of parameters will be log transformed), 'Full' (all parameters will be log transformed), and 'Custom' (Log transformed parameters specified by the customLogTransform input will be log transformed, only usable if customLogTransform was set during the creation of this modification, or modification.customLogTransform is set).
        bm : benchmarker
            A benchmarker problem to apply this modification.

        Returns
        -------
        None.

        """
        if setting.lower() == 'none':
            # Disable log transforms
            bm.log_transform([False] * bm.n_parameters())
        elif setting.lower() == 'standard':
            # Load standard log transforms from benchmarker
            bm.log_transform(bm.standardLogTransform)
        elif setting.lower() == 'full':
            # Log transform all parameters
            bm.log_transform()
        elif setting.lower() == 'custom':
            bm.log_transform(self.customLogTransform)
        else:
            print(
                "'" + setting + "' is not a valid option for log transforms. Please use either 'None', 'Standard', 'Full', or 'Custom'.")

    def apply_bounds(self, setting, bm):
        """
        Apply the bound setting to a benchmarker.

        Parameters
        ----------
        setting : string
            Setting for parameter bounds. Options are 'None' (default, none of the parameters will be bounds), 'Positive' (all parameters will have a lower bound of 0 and no upper bound), 'Sampler' (problem specific bounds using the range in the benchmarkers sampler will be used), and 'Custom' (Bounds on parameters specified by the customBounds input will be used). The default is 'None'.
        bm : benchmarker
            A benchmarker problem to apply this modification.

        Returns
        -------
        None.

        """
        if setting.lower() == 'none':
            bm.add_parameter_bounds([[-np.inf] * bm.n_parameters(), [np.inf] * bm.n_parameters()])
            bm._parameters_bounded = False
        elif setting.lower() == 'positive':
            bm.add_parameter_bounds([[0] * bm.n_parameters(), [np.inf] * bm.n_parameters()])
        elif setting.lower() == 'sampler':
            if 'staircase' in bm._name:
                bm.add_parameter_bounds([bm.defaultParams * 0.5, bm.defaultParams * 1.5])
            elif 'loewe2016' in bm._name:
                lb = []
                ub = []
                for i in range(bm.n_parameters()):
                    if bm.additiveParams[i]:
                        lb.append(bm.defaultParams[i] - 60 * bm.paramSpaceWidth)
                        ub.append(bm.defaultParams[i] + 60 * bm.paramSpaceWidth)
                    else:
                        lb.append(bm.defaultParams[i] * 10 ** -1)
                        ub.append(bm.defaultParams[i] * 10 ** 1)
                bm.add_parameter_bounds([lb, ub])
            elif 'moreno2016' in bm._name:
                bm.add_parameter_bounds(
                    [bm.defaultParams * (1 - bm.paramSpaceWidth), bm.defaultParams * (1 + bm.paramSpaceWidth)])
            elif 'test' in bm._name:
                bm.add_parameter_bounds([bm.defaultParams * 0.5, bm.defaultParams * 1.5])
            else:
                print(
                    "Could not identify the benchmarker using benchmarker._name when trying to add bounds. If a new benchmarker is being used, then this code needs updating. For the time being, you can use 'custom' bounds.")
        elif setting.lower() == 'custom':
            bm.add_parameter_bounds(self.customBounds)
        else:
            print(
                "'" + setting + "' is not a valid option for bounds. Please use either 'None', 'Positive', 'Sampler', or 'Custom'.")

    @staticmethod
    def apply_scale_factors(setting, bm):
        """
        Apply the scale factor setting to a benchmarker.

        Parameters
        ----------
        setting : string
            Setting for scale factors. Options are 'off' (default, scale factors won't be used), and 'on' (scale factors will be used, with the default parameters representing 1).
        bm : benchmarker
            A benchmarker problem to apply this modification.

        Returns
        -------
        None.

        """
        if setting.lower() == 'on':
            # Set use scale factors
            bm._useScaleFactors = True
        elif setting.lower() == 'off':
            # Disable scale factors
            bm._useScaleFactors = False
        else:
            print("'" + setting + "' is not a valid option for scale factors. Please use either 'on' or 'off'.")


class Abed2013(Modification):
    def __init__(self):
        logTransform = 'None'
        bounds = 'Sampler'  # Technically upper and lower bounds are done through a custom transformation
        scaleFactors = 'off'
        name = 'Abed2013'
        super().__init__(name=name, logTransform=logTransform, bounds=bounds, scaleFactors=scaleFactors)


class Achard2006(Modification):
    def __init__(self):
        logTransform = 'None'
        bounds = 'Sampler'
        scaleFactors = 'off'
        name = 'Achard2006'
        super().__init__(name=name, logTransform=logTransform, bounds=bounds, scaleFactors=scaleFactors)


class Balser1990(Modification):
    def __init__(self):
        logTransform = 'None'
        bounds = 'None'
        scaleFactors = 'off'
        name = 'Balser1990'
        super().__init__(name=name, logTransform=logTransform, bounds=bounds, scaleFactors=scaleFactors)


class Belletti2021(Modification):
    def __init__(self):
        logTransform = 'None'
        bounds = 'None'
        scaleFactors = 'on'
        name = 'Belletti2021'
        super().__init__(name=name, logTransform=logTransform, bounds=bounds, scaleFactors=scaleFactors)


class BenShalom2012(Modification):
    def __init__(self):
        logTransform = 'None'
        bounds = 'Sampler'
        scaleFactors = 'off'
        name = 'BenShalom2012'
        super().__init__(name=name, logTransform=logTransform, bounds=bounds, scaleFactors=scaleFactors)


class Bot2012(Modification):
    """
    The modification from Bot et al. 2012. Uses no log transforms, bounds defined by the sampler, and no scale factors.
    """

    def __init__(self):
        logTransform = 'None'
        bounds = 'Sampler'
        scaleFactors = 'on'
        name = 'Bot2012'
        super().__init__(name=name, logTransform=logTransform, bounds=bounds, scaleFactors=scaleFactors)


class BuenoOrovio2008(Modification):
    def __init__(self):
        logTransform = 'None'
        bounds = 'Sampler'
        scaleFactors = 'off'
        name = 'BuenoOrovio2008'
        super().__init__(name=name, logTransform=logTransform, bounds=bounds, scaleFactors=scaleFactors)


class Cabo2022(Modification):
    def __init__(self):
        logTransform = 'None'
        bounds = 'Sampler'
        scaleFactors = 'off'
        name = 'Cabo2022'
        super().__init__(name=name, logTransform=logTransform, bounds=bounds, scaleFactors=scaleFactors)


class Cairns2017(Modification):
    def __init__(self):
        logTransform = 'None'
        bounds = 'Sampler'
        scaleFactors = 'off'
        name = 'Cairns2017'
        super().__init__(name=name, logTransform=logTransform, bounds=bounds, scaleFactors=scaleFactors)


class Chen2012(Modification):
    """
    The modification from Chen et al. 2012. Uses sampler bounds only.
    """

    def __init__(self):
        logTransform = 'None'
        bounds = 'Sampler'
        scaleFactors = 'off'
        name = 'Chen2012'
        super().__init__(name=name, logTransform=logTransform, bounds=bounds, scaleFactors=scaleFactors)


class Clancy1999(Modification):
    def __init__(self):
        logTransform = 'None'
        bounds = 'None'
        scaleFactors = 'off'
        name = 'Clancy1999'
        super().__init__(name=name, logTransform=logTransform, bounds=bounds, scaleFactors=scaleFactors)


class Clausen2020(Modification):
    def __init__(self):
        logTransform = 'None'
        bounds = 'Sampler'  # Bounds should actually change between PSO and Nelder Mead
        scaleFactors = 'off'
        name = 'Clausen2020'
        super().__init__(name=name, logTransform=logTransform, bounds=bounds, scaleFactors=scaleFactors)


class Clerx2019(Modification):
    """
    The modification from Clerx et al. 2019. Uses standard log transforms, bounds defined by the sampler in place of rates bounds, and no scale factors.
    """

    def __init__(self):
        logTransform = 'Standard'
        bounds = 'Sampler'  # Bounds should be on rates too
        scaleFactors = 'off'
        name = 'Clerx2019'
        super().__init__(name=name, logTransform=logTransform, bounds=bounds, scaleFactors=scaleFactors)
        super().__init__(name=name, logTransform=logTransform, bounds=bounds, scaleFactors=scaleFactors,
                         kwargs={'rateBounds': True})


class Davies2011(Modification):
    def __init__(self):
        logTransform = 'None'
        bounds = 'Positive'
        scaleFactors = 'on'
        name = 'Davies2011'
        super().__init__(name=name, logTransform=logTransform, bounds=bounds, scaleFactors=scaleFactors)


class Dokos2004(Modification):
    def __init__(self):
        logTransform = 'None'
        bounds = 'Sampler'  # Technically upper and lower bounds are done through a custom transformation
        scaleFactors = 'off'
        name = 'Dokos2004'
        super().__init__(name=name, logTransform=logTransform, bounds=bounds, scaleFactors=scaleFactors)


class Druckmann2007(Modification):
    def __init__(self):
        logTransform = 'None'
        bounds = 'Sampler'
        scaleFactors = 'off'
        name = 'Druckmann2007'
        super().__init__(name=name, logTransform=logTransform, bounds=bounds, scaleFactors=scaleFactors)


class Du2014(Modification):
    def __init__(self):
        logTransform = 'None'
        bounds = 'Sampler'  # Only says constrained optimisation
        scaleFactors = 'off'
        name = 'Du2014'
        super().__init__(name=name, logTransform=logTransform, bounds=bounds, scaleFactors=scaleFactors)


class Epstein2016(Modification):
    def __init__(self):
        logTransform = 'Full'
        bounds = 'Sampler'
        scaleFactors = 'on'
        name = 'Epstein2016'
        super().__init__(name=name, logTransform=logTransform, bounds=bounds, scaleFactors=scaleFactors)


class Groenendaal2015(Modification):
    def __init__(self):
        logTransform = 'None'
        bounds = 'Sampler'
        scaleFactors = 'on'
        name = 'Groenendaal2015'
        super().__init__(name=name, logTransform=logTransform, bounds=bounds, scaleFactors=scaleFactors,
                         kwargs={'nGens': 100, 'popSize': 500})


class Guo2010(Modification):
    def __init__(self):
        logTransform = 'None'
        bounds = 'Sampler'  # Technically upper and lower bounds are done through a custom transformation
        scaleFactors = 'off'
        name = 'Guo2010'
        super().__init__(name=name, logTransform=logTransform, bounds=bounds, scaleFactors=scaleFactors)


class Gurkiewicz2007(Modification):
    def __init__(self):
        logTransform = 'None'
        bounds = 'Sampler'
        scaleFactors = 'off'
        name = 'Gurkiewicz2007'
        super().__init__(name=name, logTransform=logTransform, bounds=bounds, scaleFactors=scaleFactors)


class Hendrikson2011(Modification):
    def __init__(self):
        logTransform = 'None'
        bounds = 'Sampler'
        scaleFactors = 'off'
        name = 'Hendrikson2011'
        super().__init__(name=name, logTransform=logTransform, bounds=bounds, scaleFactors=scaleFactors)


class Houston2020(Modification):
    def __init__(self):
        logTransform = 'Standard'
        bounds = 'Sampler'
        scaleFactors = 'off'
        name = 'Houston2020'
        super().__init__(name=name, logTransform=logTransform, bounds=bounds, scaleFactors=scaleFactors)


class JedrzejewskiSzmek2018(Modification):
    def __init__(self):
        logTransform = 'None'
        bounds = 'Sampler'
        scaleFactors = 'off'
        name = 'JedrzejewskiSzmek2018'
        super().__init__(name=name, logTransform=logTransform, bounds=bounds, scaleFactors=scaleFactors,
                         kwargs={'popSize': 8})


class Kaur2014(Modification):
    def __init__(self):
        logTransform = 'None'
        bounds = 'Sampler'
        scaleFactors = 'off'
        name = 'Kaur2014'
        super().__init__(name=name, logTransform=logTransform, bounds=bounds, scaleFactors=scaleFactors)


class Kohjitani2022(Modification):
    """
    The modification from Kohjitani et al. 2022. Uses scaling factors only.
    """

    def __init__(self):
        logTransform = 'None'
        bounds = 'None'
        scaleFactors = 'on'
        name = 'Kohjitani2022'
        super().__init__(name=name, logTransform=logTransform, bounds=bounds, scaleFactors=scaleFactors)


class Liu2011(Modification):
    def __init__(self):
        logTransform = 'None'
        bounds = 'Sampler'
        scaleFactors = 'off'
        name = 'Liu2011'
        super().__init__(name=name, logTransform=logTransform, bounds=bounds, scaleFactors=scaleFactors)


class Loewe2016(Modification):
    """
    The modification from Loewe et al. 2016. Uses no log transforms, bounds defined by the sampler, and no scale factors.
    """

    def __init__(self):
        logTransform = 'None'
        bounds = 'Sampler'
        scaleFactors = 'off'
        name = 'Loewe2012'
        super().__init__(name=name, logTransform=logTransform, bounds=bounds, scaleFactors=scaleFactors)


class Maryak1998(Modification):
    def __init__(self):
        logTransform = 'None'
        bounds = 'None'
        scaleFactors = 'off'
        name = 'Maryak1998'
        super().__init__(name=name, logTransform=logTransform, bounds=bounds, scaleFactors=scaleFactors)


class Meliza2014(Modification):
    def __init__(self):
        logTransform = 'None'
        bounds = 'Sampler'
        scaleFactors = 'off'
        name = 'Meliza2014'
        super().__init__(name=name, logTransform=logTransform, bounds=bounds, scaleFactors=scaleFactors)


class Moreno2016(Modification):
    def __init__(self):
        logTransform = 'None'
        bounds = 'Positive'
        scaleFactors = 'off'
        name = 'Moreno2016'
        super().__init__(name=name, logTransform=logTransform, bounds=bounds, scaleFactors=scaleFactors)


class Munch2022(Modification):
    def __init__(self):
        logTransform = 'None'
        bounds = 'Sampler'
        scaleFactors = 'off'
        name = 'Munch2022'
        super().__init__(name=name, logTransform=logTransform, bounds=bounds, scaleFactors=scaleFactors)


class Nogaret2016(Modification):
    def __init__(self):
        logTransform = 'None'
        bounds = 'Sampler'
        scaleFactors = 'off'
        name = 'Nogaret2016'
        super().__init__(name=name, logTransform=logTransform, bounds=bounds, scaleFactors=scaleFactors)


class Nogaret2022(Modification):
    def __init__(self):
        logTransform = 'None'
        bounds = 'Sampler'
        scaleFactors = 'off'
        name = 'Nogaret2022'
        super().__init__(name=name, logTransform=logTransform, bounds=bounds, scaleFactors=scaleFactors)


class Sachse2003(Modification):
    def __init__(self):
        logTransform = 'None'
        bounds = 'None'
        scaleFactors = 'off'
        name = 'Sachse2003'
        super().__init__(name=name, logTransform=logTransform, bounds=bounds, scaleFactors=scaleFactors)


class Seemann2009(Modification):
    def __init__(self):
        logTransform = 'None'
        bounds = 'None'
        scaleFactors = 'off'
        name = 'Seemann2009'
        super().__init__(name=name, logTransform=logTransform, bounds=bounds, scaleFactors=scaleFactors)


class Smirnov2020(Modification):
    def __init__(self):
        logTransform = 'None'
        bounds = 'Sampler'
        scaleFactors = 'on'
        name = 'Smirnov2020'
        super().__init__(name=name, logTransform=logTransform, bounds=bounds, scaleFactors=scaleFactors)


class Syed2005(Modification):
    def __init__(self):
        logTransform = 'None'
        bounds = 'Sampler'
        scaleFactors = 'off'
        name = 'Syed2005'
        super().__init__(name=name, logTransform=logTransform, bounds=bounds, scaleFactors=scaleFactors)


class Taylor2020(Modification):
    def __init__(self):
        logTransform = 'None'
        bounds = 'Sampler'
        scaleFactors = 'off'
        name = 'Taylor2020'
        super().__init__(name=name, logTransform=logTransform, bounds=bounds, scaleFactors=scaleFactors)


class Vanier1999(Modification):
    def __init__(self):
        logTransform = 'None'
        bounds = 'Sampler'  # Finite volume search space for one approach. Wasn't clear what this search space is
        scaleFactors = 'on'  # Its only actually on some parameter here
        name = 'Vanier1999'
        super().__init__(name=name, logTransform=logTransform, bounds=bounds, scaleFactors=scaleFactors)


class Vavoulis2012(Modification):
    def __init__(self):
        logTransform = 'Full'
        bounds = 'Sampler'
        scaleFactors = 'on'
        name = 'Vavoulis2012'
        super().__init__(name=name, logTransform=logTransform, bounds=bounds, scaleFactors=scaleFactors)


class Weber2008(Modification):
    def __init__(self):
        logTransform = 'None'
        bounds = 'None'
        scaleFactors = 'off'
        name = 'Weber2008'
        super().__init__(name=name, logTransform=logTransform, bounds=bounds, scaleFactors=scaleFactors)


class Wilhelms2012a(Modification):
    def __init__(self):
        logTransform = 'None'
        bounds = 'None'
        scaleFactors = 'off'
        name = 'Wilhelms2012a'
        super().__init__(name=name, logTransform=logTransform, bounds=bounds, scaleFactors=scaleFactors)


class Wilhelms2012b(Modification):
    def __init__(self):
        logTransform = 'None'
        bounds = 'Sampler'
        scaleFactors = 'off'
        name = 'Wilhelms2012b'
        super().__init__(name=name, logTransform=logTransform, bounds=bounds, scaleFactors=scaleFactors)


class Zhou2009(Modification):
    def __init__(self):
        logTransform = 'None'
        bounds = 'None'
        scaleFactors = 'off'
        name = 'Zhou2009'
        super().__init__(name=name, logTransform=logTransform, bounds=bounds, scaleFactors=scaleFactors)


class Empty(Modification):
    """
    A modification with default settings.
    """

    def __init__(self, name=""):
        logTransform = 'None'
        bounds = 'None'
        scaleFactors = 'off'
        super().__init__(name=name, logTransform=logTransform, bounds=bounds, scaleFactors=scaleFactors)

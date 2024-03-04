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

    def __init__(self, name="", logTransform='off', parameterBounds='off', rateBounds='off', scaleFactors='off',
                 customLogTransform=None,
                 customBounds=None, kwargs=None):
        """
        Initialise a Modification object. Once a Modification is built, its setting can be applied to any benchmarker, bm, by calling modification.apply(bm)

        Parameters
        ----------
        name : string, optional
            A name for the modification. Useful for logging. The default is "".
        logTransform : string, optional
            Setting for log transforms. Options are 'off' (default, none of the parameters will be log transformed), 'on' (a problem specific set of parameters will be log transformed), and 'Custom' (Log transformed parameters specified by the customLogTransform input will be log transformed).
        parameterBounds : string, optional
            Setting for parameter bounds. Options are 'off' (default, none of the parameters will be bounds), 'on' (problem specific bounds using the range in the benchmarkers sampler will be used), and 'Custom' (Bounds on parameters specified by the customBounds input will be used).
        rateBounds : string, optional
            Setting for rate bounds on parameter combinations. Options are 'off' (default, rate bounds will not be used), 'on' (rate bounds will be used).
        scaleFactors : string, optional
            Setting for scale factors. Options are 'off' (default, scale factors won't be used), and 'on' (scale factors will be used, with the default parameters representing 1).
        customLogTransform : list, optional
            A list of parameters to log transform. The list should be the same length as the number of parameters in the benchmarker. This makes this option problem specific and will be unlikely to be usable across different benchmarkers. Only required if logTransform = 'Custom'.
        customBounds : list, optional
            A list containing 2 elements, a list of lower bounds and a list of upper bounds. Each sub-list should be the same length as the number of parameters in the benchmarker. This makes this option problem specific and will be unlikely to be usable across different benchmarkers. Only required if parameterBounds = 'Custom'.
        kwargs : dict, optional
            A dictionary of keyword arguments to be passed into an optimisers run function. The default is None.
        """
        if kwargs is None:
            kwargs = {}
        self.dict = {'log transform': logTransform, 'parameterBounds': parameterBounds, 'rateBounds': rateBounds,
                     'scale factors': scaleFactors}
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
        """
        self.apply_log_transforms(self.dict['log transform'], bm)
        self.apply_parameter_bounds(self.dict['parameterBounds'], bm)
        self.apply_rate_bounds(self.dict['rateBounds'], bm)
        self.apply_scale_factors(self.dict['scale factors'], bm)

    def apply_log_transforms(self, setting, bm):
        """
        Apply the log transform setting to a benchmarker.

        Parameters
        ----------
        setting : string
            Setting for log transforms. Options are 'off' (default, none of the parameters will be log transformed), 'on' (a problem specific set of parameters will be log transformed), and 'Custom' (Log transformed parameters specified by the customLogTransform input will be log transformed, only usable if customLogTransform was set during the creation of this modification, or modification.customLogTransform is set).
        bm : benchmarker
            A benchmarker problem to apply this modification.
        """
        if setting.lower() == 'off':
            # Disable log transforms
            bm.log_transform([False] * bm.n_parameters())
        elif setting.lower() == 'on':
            # Load standard log transforms from benchmarker
            bm.log_transform(bm.standardLogTransform)
        elif setting.lower() == 'custom':
            bm.log_transform(self.customLogTransform)
        else:
            print(
                "'" + setting + "' is not a valid option for log transforms. Please use either 'off', 'on', or 'Custom'.")

    def apply_parameter_bounds(self, setting, bm):
        """
        Apply the parameter bounds setting to a benchmarker.

        Parameters
        ----------
        setting : string
            Setting for parameter bounds. Options are 'off' (default, none of the parameters will be bounds), 'on' (problem specific bounds using the range in the benchmarkers sampler will be used), and 'Custom' (Bounds on parameters specified by the customBounds input will be used).
        bm : benchmarker
            A benchmarker problem to apply this modification.
        """
        if setting.lower() == 'off':
            bm._parameters_bounded = False
        elif setting.lower() == 'on':
            bm.add_parameter_bounds()
        elif setting.lower() == 'custom':
            bm.add_parameter_bounds()
            bm.lb = self.customBounds[0]
            bm.ub = self.customBounds[1]
        else:
            print(
                "'" + setting + "' is not a valid option for bounds. Please use either 'off', 'on', or 'Custom'.")

    @staticmethod
    def apply_rate_bounds(setting, bm):
        """
        Apply the rate bounds setting to a benchmarker.

        Parameters
        ----------
        setting : string
            Setting for rate bounds. Options are 'off' (default, rate bounds will not be used), and 'on' (rate bounds will be used).
        bm : benchmarker
            A benchmarker problem to apply this modification.
        """
        if setting.lower() == 'off':
            bm._rates_bounded = False
        elif setting.lower() == 'on':
            bm.add_rate_bounds()
        else:
            print(
                "'" + setting + "' is not a valid option for bounds. Please use either 'off', or 'on'.")

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
        logTransform = 'off'
        parameterBounds = 'on'  # Technically upper and lower bounds are done through a custom transformation
        scaleFactors = 'off'
        name = 'Abed2013'
        super().__init__(name=name, logTransform=logTransform, parameterBounds=parameterBounds, scaleFactors=scaleFactors)


class Achard2006(Modification):
    def __init__(self):
        logTransform = 'off'
        parameterBounds = 'on'
        scaleFactors = 'off'
        name = 'Achard2006'
        super().__init__(name=name, logTransform=logTransform, parameterBounds=parameterBounds, scaleFactors=scaleFactors)


class Balser1990(Modification):
    def __init__(self):
        logTransform = 'off'
        parameterBounds = 'off'
        scaleFactors = 'off'
        name = 'Balser1990'
        super().__init__(name=name, logTransform=logTransform, parameterBounds=parameterBounds, scaleFactors=scaleFactors)


class Belletti2021(Modification):
    def __init__(self):
        logTransform = 'off'
        parameterBounds = 'off'
        scaleFactors = 'on'
        name = 'Belletti2021'
        super().__init__(name=name, logTransform=logTransform, parameterBounds=parameterBounds, scaleFactors=scaleFactors)


class BenShalom2012(Modification):
    def __init__(self):
        logTransform = 'off'
        parameterBounds = 'on'
        scaleFactors = 'off'
        name = 'BenShalom2012'
        super().__init__(name=name, logTransform=logTransform, parameterBounds=parameterBounds, scaleFactors=scaleFactors)


class Bot2012(Modification):
    def __init__(self):
        logTransform = 'off'
        parameterBounds = 'on'
        scaleFactors = 'on'
        name = 'Bot2012'
        super().__init__(name=name, logTransform=logTransform, parameterBounds=parameterBounds, scaleFactors=scaleFactors)


class BuenoOrovio2008(Modification):
    def __init__(self):
        logTransform = 'off'
        parameterBounds = 'on'
        scaleFactors = 'off'
        name = 'BuenoOrovio2008'
        super().__init__(name=name, logTransform=logTransform, parameterBounds=parameterBounds, scaleFactors=scaleFactors)


class Cabo2022(Modification):
    def __init__(self):
        logTransform = 'off'
        parameterBounds = 'on'
        scaleFactors = 'off'
        name = 'Cabo2022'
        super().__init__(name=name, logTransform=logTransform, parameterBounds=parameterBounds, scaleFactors=scaleFactors)


class Cairns2017(Modification):
    def __init__(self):
        logTransform = 'off'
        parameterBounds = 'on'
        scaleFactors = 'off'
        name = 'Cairns2017'
        super().__init__(name=name, logTransform=logTransform, parameterBounds=parameterBounds, scaleFactors=scaleFactors)


class Chen2012(Modification):
    def __init__(self):
        logTransform = 'off'
        parameterBounds = 'on'
        scaleFactors = 'off'
        name = 'Chen2012'
        super().__init__(name=name, logTransform=logTransform, parameterBounds=parameterBounds, scaleFactors=scaleFactors)


class Clancy1999(Modification):
    def __init__(self):
        logTransform = 'off'
        parameterBounds = 'off'
        scaleFactors = 'off'
        name = 'Clancy1999'
        super().__init__(name=name, logTransform=logTransform, parameterBounds=parameterBounds, scaleFactors=scaleFactors)


class Clausen2020(Modification):
    def __init__(self):
        logTransform = 'off'
        parameterBounds = 'on'  # Bounds should actually change between PSO and Nelder Mead
        scaleFactors = 'off'
        name = 'Clausen2020'
        super().__init__(name=name, logTransform=logTransform, parameterBounds=parameterBounds, scaleFactors=scaleFactors)


class Clerx2019(Modification):
    def __init__(self):
        logTransform = 'on'
        parameterBounds = 'on'
        rateBounds = 'on'
        scaleFactors = 'off'
        name = 'Clerx2019'
        super().__init__(name=name, logTransform=logTransform, parameterBounds=parameterBounds, rateBounds=rateBounds, scaleFactors=scaleFactors)


class Davies2011(Modification):
    def __init__(self):
        logTransform = 'off'
        parameterBounds = 'on'
        scaleFactors = 'on'
        name = 'Davies2011'
        super().__init__(name=name, logTransform=logTransform, parameterBounds=parameterBounds, scaleFactors=scaleFactors)


class Dokos2004(Modification):
    def __init__(self):
        logTransform = 'off'
        parameterBounds = 'on'  # Technically upper and lower bounds are done through a custom transformation
        scaleFactors = 'off'
        name = 'Dokos2004'
        super().__init__(name=name, logTransform=logTransform, parameterBounds=parameterBounds, scaleFactors=scaleFactors)


class Druckmann2007(Modification):
    def __init__(self):
        logTransform = 'off'
        parameterBounds = 'on'
        scaleFactors = 'off'
        name = 'Druckmann2007'
        super().__init__(name=name, logTransform=logTransform, parameterBounds=parameterBounds, scaleFactors=scaleFactors)


class Du2014(Modification):
    def __init__(self):
        logTransform = 'off'
        parameterBounds = 'on'
        scaleFactors = 'off'
        name = 'Du2014'
        super().__init__(name=name, logTransform=logTransform, parameterBounds=parameterBounds, scaleFactors=scaleFactors)


class Epstein2016(Modification):
    def __init__(self):
        logTransform = 'on'
        parameterBounds = 'on'
        scaleFactors = 'on'
        name = 'Epstein2016'
        super().__init__(name=name, logTransform=logTransform, parameterBounds=parameterBounds, scaleFactors=scaleFactors)


class Groenendaal2015(Modification):
    def __init__(self):
        logTransform = 'off'
        parameterBounds = 'on'
        scaleFactors = 'on'
        name = 'Groenendaal2015'
        super().__init__(name=name, logTransform=logTransform, parameterBounds=parameterBounds, scaleFactors=scaleFactors,
                         kwargs={'nGens': 100, 'popSize': 500})


class Guo2010(Modification):
    def __init__(self):
        logTransform = 'off'
        parameterBounds = 'on'  # Technically upper and lower bounds are done through a custom transformation
        scaleFactors = 'off'
        name = 'Guo2010'
        super().__init__(name=name, logTransform=logTransform, parameterBounds=parameterBounds, scaleFactors=scaleFactors)


class Gurkiewicz2007(Modification):
    def __init__(self):
        logTransform = 'off'
        parameterBounds = 'on'
        scaleFactors = 'off'
        name = 'Gurkiewicz2007'
        super().__init__(name=name, logTransform=logTransform, parameterBounds=parameterBounds, scaleFactors=scaleFactors)


class Hendrikson2011(Modification):
    def __init__(self):
        logTransform = 'off'
        parameterBounds = 'on'
        scaleFactors = 'off'
        name = 'Hendrikson2011'
        super().__init__(name=name, logTransform=logTransform, parameterBounds=parameterBounds, scaleFactors=scaleFactors)


class Houston2020(Modification):
    def __init__(self):
        logTransform = 'on'
        parameterBounds = 'on'
        scaleFactors = 'off'
        name = 'Houston2020'
        super().__init__(name=name, logTransform=logTransform, parameterBounds=parameterBounds, scaleFactors=scaleFactors)


class JedrzejewskiSzmek2018(Modification):
    def __init__(self):
        logTransform = 'off'
        parameterBounds = 'on'
        scaleFactors = 'off'
        name = 'JedrzejewskiSzmek2018'
        super().__init__(name=name, logTransform=logTransform, parameterBounds=parameterBounds, scaleFactors=scaleFactors,
                         kwargs={'popSize': 8})


class Kaur2014(Modification):
    def __init__(self):
        logTransform = 'off'
        parameterBounds = 'on'
        scaleFactors = 'off'
        name = 'Kaur2014'
        super().__init__(name=name, logTransform=logTransform, parameterBounds=parameterBounds, scaleFactors=scaleFactors)


class Kohjitani2022(Modification):
    def __init__(self):
        logTransform = 'off'
        parameterBounds = 'off'
        scaleFactors = 'on'
        name = 'Kohjitani2022'
        super().__init__(name=name, logTransform=logTransform, parameterBounds=parameterBounds, scaleFactors=scaleFactors)


class Liu2011(Modification):
    def __init__(self):
        logTransform = 'off'
        parameterBounds = 'on'
        scaleFactors = 'off'
        name = 'Liu2011'
        super().__init__(name=name, logTransform=logTransform, parameterBounds=parameterBounds, scaleFactors=scaleFactors)


class Loewe2016(Modification):
    def __init__(self):
        logTransform = 'off'
        parameterBounds = 'on'
        scaleFactors = 'off'
        name = 'Loewe2012'
        super().__init__(name=name, logTransform=logTransform, parameterBounds=parameterBounds, scaleFactors=scaleFactors)


class Maryak1998(Modification):
    def __init__(self):
        logTransform = 'off'
        parameterBounds = 'off'
        scaleFactors = 'off'
        name = 'Maryak1998'
        super().__init__(name=name, logTransform=logTransform, parameterBounds=parameterBounds, scaleFactors=scaleFactors)


class Meliza2014(Modification):
    def __init__(self):
        logTransform = 'off'
        parameterBounds = 'on'
        scaleFactors = 'off'
        name = 'Meliza2014'
        super().__init__(name=name, logTransform=logTransform, parameterBounds=parameterBounds, scaleFactors=scaleFactors)


class Moreno2016(Modification):
    def __init__(self):
        logTransform = 'off'
        parameterBounds = 'on'
        scaleFactors = 'off'
        name = 'Moreno2016'
        super().__init__(name=name, logTransform=logTransform, parameterBounds=parameterBounds, scaleFactors=scaleFactors)


class Munch2022(Modification):
    def __init__(self):
        logTransform = 'off'
        parameterBounds = 'on'
        scaleFactors = 'off'
        name = 'Munch2022'
        super().__init__(name=name, logTransform=logTransform, parameterBounds=parameterBounds, scaleFactors=scaleFactors)


class Nogaret2016(Modification):
    def __init__(self):
        logTransform = 'off'
        parameterBounds = 'on'
        scaleFactors = 'off'
        name = 'Nogaret2016'
        super().__init__(name=name, logTransform=logTransform, parameterBounds=parameterBounds, scaleFactors=scaleFactors)


class Nogaret2022(Modification):
    def __init__(self):
        logTransform = 'off'
        parameterBounds = 'on'
        scaleFactors = 'off'
        name = 'Nogaret2022'
        super().__init__(name=name, logTransform=logTransform, parameterBounds=parameterBounds, scaleFactors=scaleFactors)


class Sachse2003(Modification):
    def __init__(self):
        logTransform = 'off'
        parameterBounds = 'off'
        scaleFactors = 'off'
        name = 'Sachse2003'
        super().__init__(name=name, logTransform=logTransform, parameterBounds=parameterBounds, scaleFactors=scaleFactors)


class Seemann2009(Modification):
    def __init__(self):
        logTransform = 'off'
        parameterBounds = 'off'
        scaleFactors = 'off'
        name = 'Seemann2009'
        super().__init__(name=name, logTransform=logTransform, parameterBounds=parameterBounds, scaleFactors=scaleFactors)


class Smirnov2020(Modification):
    def __init__(self):
        logTransform = 'off'
        parameterBounds = 'on'
        scaleFactors = 'on'
        name = 'Smirnov2020'
        super().__init__(name=name, logTransform=logTransform, parameterBounds=parameterBounds, scaleFactors=scaleFactors)


class Syed2005(Modification):
    def __init__(self):
        logTransform = 'off'
        parameterBounds = 'on'
        scaleFactors = 'off'
        name = 'Syed2005'
        super().__init__(name=name, logTransform=logTransform, parameterBounds=parameterBounds, scaleFactors=scaleFactors)


class Taylor2020(Modification):
    def __init__(self):
        logTransform = 'off'
        parameterBounds = 'on'
        scaleFactors = 'off'
        name = 'Taylor2020'
        super().__init__(name=name, logTransform=logTransform, parameterBounds=parameterBounds, scaleFactors=scaleFactors)


class Vanier1999(Modification):
    def __init__(self):
        logTransform = 'off'
        parameterBounds = 'on'
        scaleFactors = 'on'
        name = 'Vanier1999'
        super().__init__(name=name, logTransform=logTransform, parameterBounds=parameterBounds, scaleFactors=scaleFactors)


class Vavoulis2012(Modification):
    def __init__(self):
        logTransform = 'on'
        parameterBounds = 'on'
        scaleFactors = 'on'
        name = 'Vavoulis2012'
        super().__init__(name=name, logTransform=logTransform, parameterBounds=parameterBounds, scaleFactors=scaleFactors)


class Weber2008(Modification):
    def __init__(self):
        logTransform = 'off'
        parameterBounds = 'off'
        scaleFactors = 'off'
        name = 'Weber2008'
        super().__init__(name=name, logTransform=logTransform, parameterBounds=parameterBounds, scaleFactors=scaleFactors)


class Wilhelms2012a(Modification):
    def __init__(self):
        logTransform = 'off'
        parameterBounds = 'off'
        scaleFactors = 'off'
        name = 'Wilhelms2012a'
        super().__init__(name=name, logTransform=logTransform, parameterBounds=parameterBounds, scaleFactors=scaleFactors)


class Wilhelms2012b(Modification):
    def __init__(self):
        logTransform = 'off'
        parameterBounds = 'on'
        scaleFactors = 'off'
        name = 'Wilhelms2012b'
        super().__init__(name=name, logTransform=logTransform, parameterBounds=parameterBounds, scaleFactors=scaleFactors)


class Zhou2009(Modification):
    def __init__(self):
        logTransform = 'off'
        parameterBounds = 'off'
        scaleFactors = 'off'
        name = 'Zhou2009'
        super().__init__(name=name, logTransform=logTransform, parameterBounds=parameterBounds, scaleFactors=scaleFactors)


class Empty(Modification):
    """
    A modification with default settings.
    """
    def __init__(self, name=""):
        logTransform = 'off'
        parameterBounds = 'off'
        scaleFactors = 'off'
        super().__init__(name=name, logTransform=logTransform, parameterBounds=parameterBounds, scaleFactors=scaleFactors)

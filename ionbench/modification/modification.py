"""
Contains the main Modification class and all the paper-specific modification subclasses.
"""


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
        # Store a dictionary of settings
        self.dict = {'log transform': logTransform, 'parameterBounds': parameterBounds, 'rateBounds': rateBounds,
                     'scale factors': scaleFactors}
        self.customLogTransform = customLogTransform
        self.customBounds = customBounds
        if len(name) > 0:
            self.NAME = name
        else:
            self.NAME = None
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
            bm.log_transform(bm.STANDARD_LOG_TRANSFORM)
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
            bm.parametersBounded = False
        elif setting.lower() == 'on':
            bm.add_parameter_bounds()
        elif setting.lower() == 'custom':
            bm.add_parameter_bounds()
            bm.lb = self.customBounds[0]
            bm.ub = self.customBounds[1]
        else:
            print(
                "'" + setting + "' is not a valid option for bounds. Please use either 'off', 'on', or 'Custom'.")

        # If staircase, override with True
        if 'staircase' in bm.NAME:
            bm.add_parameter_bounds()

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
            bm.ratesBounded = False
        elif setting.lower() == 'on':
            bm.add_rate_bounds()
        else:
            print(
                "'" + setting + "' is not a valid option for bounds. Please use either 'off', or 'on'.")

        # If staircase, override with True
        if 'staircase' in bm.NAME:
            bm.add_rate_bounds()

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
            bm.useScaleFactors = True
        elif setting.lower() == 'off':
            # Disable scale factors
            bm.useScaleFactors = False
        else:
            print("'" + setting + "' is not a valid option for scale factors. Please use either 'on' or 'off'.")


class Abed2013(Modification):
    def __init__(self):
        # Technically upper and lower bounds are done through a custom transformation
        super().__init__(name='Abed2013', logTransform='off', parameterBounds='on', scaleFactors='off')


class Balser1990(Modification):
    def __init__(self):
        super().__init__(name='Balser1990', logTransform='off', parameterBounds='off', scaleFactors='off')


class Beattie2018(Modification):
    def __init__(self):
        super().__init__(name='Beattie2018', logTransform='on', parameterBounds='on', rateBounds='on', scaleFactors='off')


class BenShalom2012(Modification):
    def __init__(self):
        super().__init__(name='BenShalom2012', logTransform='off', parameterBounds='on', scaleFactors='off', kwargs={'crossoverProb': 0.1})


class Bot2012(Modification):
    def __init__(self):
        super().__init__(name='Bot2012', logTransform='off', parameterBounds='on', scaleFactors='on')


class BuenoOrovio2008(Modification):
    def __init__(self):
        super().__init__(name='BuenoOrovio2008', logTransform='off', parameterBounds='on', scaleFactors='off')


class Cabo2022(Modification):
    def __init__(self):
        super().__init__(name='Cabo2022', logTransform='off', parameterBounds='on', scaleFactors='off')


class Cairns2017(Modification):
    def __init__(self):
        super().__init__(name='Cairns2017', logTransform='off', parameterBounds='on', scaleFactors='off')


class Chen2012(Modification):
    def __init__(self):
        super().__init__(name='Chen2012', logTransform='off', parameterBounds='on', scaleFactors='off')


class Clancy1999(Modification):
    def __init__(self):
        super().__init__(name='Clancy1999', logTransform='off', parameterBounds='off', scaleFactors='off')


class Clausen2020(Modification):
    def __init__(self):
        # Bounds should actually change between PSO and Nelder Mead
        super().__init__(name='Clausen2020', logTransform='off', parameterBounds='on', scaleFactors='off')


class Clerx2019(Modification):
    def __init__(self):
        super().__init__(name='Clerx2019', logTransform='on', parameterBounds='on', rateBounds='on', scaleFactors='off')


class Davies2012(Modification):
    def __init__(self):
        super().__init__(name='Davies2012', logTransform='off', parameterBounds='on', scaleFactors='on')


class Dokos2004(Modification):
    def __init__(self):
        # Technically upper and lower bounds are done through a custom transformation
        super().__init__(name='Dokos2004', logTransform='off', parameterBounds='on', scaleFactors='off')


class Du2014(Modification):
    def __init__(self):
        super().__init__(name='Du2014', logTransform='off', parameterBounds='on', scaleFactors='off')


class Groenendaal2015(Modification):
    def __init__(self):
        super().__init__(name='Groenendaal2015', logTransform='off', parameterBounds='on', scaleFactors='on',
                         kwargs={'nGens': 100, 'popSize': 500})


class Guo2010(Modification):
    def __init__(self):
        # Technically upper and lower bounds are done through a custom transformation
        super().__init__(name='Guo2010', logTransform='off', parameterBounds='on', scaleFactors='off')


class Gurkiewicz2007(Modification):
    def __init__(self):
        super().__init__(name='Gurkiewicz2007', logTransform='off', parameterBounds='on', scaleFactors='off')


class JedrzejewskiSzmek2018(Modification):
    def __init__(self):
        super().__init__(name='JedrzejewskiSzmek2018', logTransform='off', parameterBounds='on', scaleFactors='off',
                         kwargs={'popSize': 8})


class Kohjitani2022(Modification):
    def __init__(self):
        super().__init__(name='Kohjitani2022', logTransform='off', parameterBounds='off', scaleFactors='on')


class Liu2011(Modification):
    def __init__(self):
        super().__init__(name='Liu2011', logTransform='off', parameterBounds='on', scaleFactors='off')


class Loewe2016(Modification):
    def __init__(self):
        super().__init__(name='Loewe2016', logTransform='off', parameterBounds='on', scaleFactors='off')


class Maryak1998(Modification):
    def __init__(self):
        super().__init__(name='Maryak1998', logTransform='off', parameterBounds='off', scaleFactors='off')


class Moreno2016(Modification):
    def __init__(self):
        super().__init__(name='Moreno2016', logTransform='off', parameterBounds='on', scaleFactors='off')


class Sachse2003(Modification):
    def __init__(self):
        super().__init__(name='Sachse2003', logTransform='off', parameterBounds='off', scaleFactors='off')


class Seemann2009(Modification):
    def __init__(self):
        super().__init__(name='Seemann2009', logTransform='off', parameterBounds='off', scaleFactors='off')


class Smirnov2020(Modification):
    def __init__(self):
        super().__init__(name='Smirnov2020', logTransform='off', parameterBounds='on', scaleFactors='on')


class Vanier1999(Modification):
    def __init__(self):
        super().__init__(name='Vanier1999', logTransform='off', parameterBounds='on', scaleFactors='on')


class Wilhelms2012a(Modification):
    def __init__(self):
        super().__init__(name='Wilhelms2012a', logTransform='off', parameterBounds='off', scaleFactors='off')


class Wilhelms2012b(Modification):
    def __init__(self):
        super().__init__(name='Wilhelms2012b', logTransform='off', parameterBounds='on', scaleFactors='off')


class Zhou2009(Modification):
    def __init__(self):
        super().__init__(name='Zhou2009', logTransform='off', parameterBounds='off', scaleFactors='off')


class Empty(Modification):
    """
    A modification with default settings.
    """

    def __init__(self, name=""):
        super().__init__(name=name, logTransform='off', parameterBounds='off', scaleFactors='off')

import ionbench
import numpy as np

class Approach():
    """
    Approaches provide a way to store some benchmarker settings and apply them to a range of benchmarkers. These settings include log-transforms, parameter bounds, and the use of scale factors. A single approach can be generated, and then applied to a range of benchmarkers. Once a Approach object is instantiated, its transforms and bounds can be applied to a benchmarker, bm, by calling approach.apply(bm)
    """
    def __init__(self, logTransform = 'None', bounds = 'None', scaleFactors = 'off', customLogTransform = None, customBounds = None):
        """
        Initialise an Approach object. Once an Approach is built, its setting can be applied to any benchmarker, bm, by calling approach.apply(bm)

        Parameters
        ----------
        logTransform : string, optional
            Setting for log transforms. Options are 'None' (default, none of the parameters will be log transformed), 'Standard' (a problem specific set of parameters will be log transformed), 'Full' (all parameters will be log transformed), and 'Custom' (Log transformed parameters specified by the customLogTransform input will be log transformed).
        bounds : string, optional
            Setting for parameter bounds. Options are 'None' (default, none of the parameters will be bounds), 'Positive' (all parameters will have a lower bound of 0 and no upper bound), 'Sampler' (problem specific bounds using the range in the benchmarkers sampler will be used), and 'Custom' (Bounds on parameters specified by the customBounds input will be used). The default is 'None'.
        scaleFactors : string, optional
            Setting for scale factors. Options are 'off' (default, scale factors won't be used), and 'on' (scale factors will be used, with the default parameters representing 1).
        customLogTransform : list, optional
            A list of parameters to log transform. The list should be the same length as the number of parameters in the benchmarker. This makes this option problem specific and will be unlikely to be useable across different benchmarkers. Only required if logTransform = 'Custom'. 
        customBounds : list, optional
            A list containing 2 elements, a list of lower bounds and a list of upper bounds. Each sub-list should be the same length as the number of parameters in the benchmarker. This makes this option problem specific and will be unlikely to be useable across different benchmarkers. Only required if bounds = 'Custom'. 

        Returns
        -------
        None.

        """
        self.dict = {'log transform':logTransform, 'bounds':bounds, 'scale factors':scaleFactors}
        self.customLogTransform = customLogTransform
        self.customBounds = customBounds
    
    def apply(self, bm):
        """
        Applies the settings in this approach to the inputted benchmarker. This will override any previous settings assigned to the benchmarker.

        Parameters
        ----------
        bm : benchmarker
            A benchmarker problem to apply this approach.

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
            Setting for log transforms. Options are 'None' (default, none of the parameters will be log transformed), 'Standard' (a problem specific set of parameters will be log transformed), 'Full' (all parameters will be log transformed), and 'Custom' (Log transformed parameters specified by the customLogTransform input will be log transformed, only usable if customLogTransform was set during the creation of this approach, or approach.customLogTransform is set).
        bm : benchmarker
            A benchmarker problem to apply this approach.

        Returns
        -------
        None.

        """
        if setting.lower() == 'none':
            #Disable log transforms
            bm.log_transform([False]*bm.n_parameters())
        elif setting.lower() == 'standard':
            #Load standard log transforms from benchmarker
            bm.log_transform(bm.standardLogTransform)
        elif setting.lower() == 'full':
            #Log transform all parameters
            bm.log_transform()
        elif setting.lower() == 'custom':
            bm.log_transform(self.customLogTransform)
        else:
            print("'"+setting+"' is not a valid option for log transforms. Please use either 'None', 'Standard', 'Full', or 'Custom'.")
    
    def apply_bounds(self, setting, bm):
        """
        Apply the bound setting to a benchmarker.

        Parameters
        ----------
        setting : string
            Setting for parameter bounds. Options are 'None' (default, none of the parameters will be bounds), 'Positive' (all parameters will have a lower bound of 0 and no upper bound), 'Sampler' (problem specific bounds using the range in the benchmarkers sampler will be used), and 'Custom' (Bounds on parameters specified by the customBounds input will be used). The default is 'None'.
        bm : benchmarker
            A benchmarker problem to apply this approach.

        Returns
        -------
        None.

        """
        if setting.lower() == 'none':
            bm.add_bounds([[-np.inf]*bm.n_parameters(),[np.inf]*bm.n_parameters()])
            bm._bounded = False
        elif setting.lower() == 'positive':
            bm.add_bounds([[0]*bm.n_parameters(),[np.inf]*bm.n_parameters()])
        elif setting.lower() == 'sampler':
            if 'staircase' in bm._name:
                bm.add_bounds([bm.defaultParams*0.5,bm.defaultParams*1.5])
            elif 'loewe2016' in bm._name:
                lb = []
                ub = []
                for i in range(bm.n_parameters()):
                    if bm.additiveParams[i]:
                        lb.append(bm.defaultParams[i] - 60*bm.paramSpaceWidth)
                        ub.append(bm.defaultParams[i] + 60*bm.paramSpaceWidth)
                    else:
                        lb.append(bm.defaultParams[i]*10**-1)
                        ub.append(bm.defaultParams[i]*10**1)
                bm.add_bounds([lb,ub])
            elif 'moreno2016' in bm._name:
                bm.add_bounds([bm.defaultParams*(1-bm.paramSpaceWidth), bm.defaultParams*(1+bm.paramSpaceWidth)])
            else:
                print("Could not identify the benchmaker using benchmarker._name when trying to add bounds. If a new benchmarker is being used, then this code needs updating. For the time being, you can use 'custom' bounds.")
        elif setting.lower() == 'custom':
            bm.add_bounds(self.customBounds)
        else:
            print("'"+setting+"' is not a valid option for bounds. Please use either 'None', 'Positive', 'Sampler', or 'Custom'.")
    
    def apply_scale_factors(self, setting, bm):
        """
        Apply the scale factor setting to a benchmarker.

        Parameters
        ----------
        setting : string
            Setting for scale factors. Options are 'off' (default, scale factors won't be used), and 'on' (scale factors will be used, with the default parameters representing 1).
        bm : benchmarker
            A benchmarker problem to apply this approach.

        Returns
        -------
        None.

        """
        if setting.lower() == 'on':
            #Set use scale factors
            bm._useScaleFactors = True
        elif setting.lower() == 'off':
            #Disable scale factors
            bm._useScaleFactors = False
        else:
            print("'"+setting+"' is not a valid option for scale factors. Please use either 'on' or 'off'.")

class Clerx2019(Approach):
    """
    The approach from Clerx et al 2019. Uses standard log transforms, bounds defined by the sampler in place of rates bounds, and no scale factors.
    """
    def __init__(self):
        logTransform = 'Standard'
        bounds = 'Sampler'
        scaleFactors = 'off'
        super().__init__(logTransform = logTransform, bounds = bounds, scaleFactors = scaleFactors)

class Loewe2016(Approach):
    """
    The approach from Loewe et al 2016. Uses no log transforms, bounds defined by the sampler, and no scale factors.
    """
    def __init__(self):
        logTransform = 'None'
        bounds = 'Sampler'
        scaleFactors = 'off'
        super().__init__(logTransform = logTransform, bounds = bounds, scaleFactors = scaleFactors)

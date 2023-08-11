import pints
import numpy as np

class Model(pints.ForwardModel):
    """
    A Pints forwards model containing a benchmarker class.
    """
    def __init__(self, bm):
        self.bm = bm
        
    def n_parameters(self):
        return self.bm.n_parameters()
    
    def simulate(self, parameters, times):
        # Reset the simulation
        return self.bm.simulate(parameters, times)

class AdvancedBoundaries(pints.Boundaries):
    """
    Pints boundaries to apply to the parameters and the rates. Currently the rates are hard-coded to correspond to the Hodgkin-Huxley IKr staircase benchmarker.
    """
    def __init__(self, paramCount, localBounds, kCombinations, bm, vHigh = 40, vLow = -120):
        """
        Build a Pints boundary object to apply parameter and rate bounds.

        Parameters
        ----------
        paramCount : int
            Number of parameters in the model.
        localBounds : list
            Bounds on the parameters are specified in localBounds. For example, localBounds = [[0,1e-7,1e3],[3,1e-3,1e5]] sets the bounds for parameter index 0 to be [1e-7,1e3] and index 3 to be [1e-3,1e5].
        kCombinations : list
            kCombinations = [[0,1],[4,5]] means param[0]*exp(param[1]*V) and param[4]*exp(param[5]*V) satisfy bounds.
        bm : Benchmarker
            A test problem benchmarker.
        vHigh : float, optional
            The high voltage to use for the rate bounds. The default is 40.
        vLow : float, optional
            The low voltage to use for the rate bounds. The default is -120.

        Returns
        -------
        None.

        """
        # localBounds = [[0,1e-7,1e3],[3,1e-3,1e5]] sets the bounds for parameter index 0 to be [1e-7,1e3] and index 3 to be [1e-3,1e5]
        # kCombinations = [[0,1],[4,5]] means param[0]*exp(param[1]*V) and param[4]*exp(param[5]*V) satisfy bounds
        # self.a_min = 1e-7
        # self.a_max = 1e3
        # self.b_min = 1e-7
        # self.b_max = 0.4
        self.bm = bm
        self.km_min = 1.67e-5
        self.km_max = 1e3
        self.vLow = vLow
        self.vHigh = vHigh
        self.paramCount = paramCount
        self.kCombinations = kCombinations
        
        self.lowerBounds = [-np.inf for i in range(paramCount)]
        self.upperBounds = [np.inf for i in range(paramCount)]
        for bound in localBounds:
            self.lowerBounds[bound[0]] = bound[1]
            self.upperBounds[bound[0]] = bound[2]
    
    def n_parameters(self):
        """
        Returns the number of parameters in the model.
        """
        return self.paramCount
    
    def check(self, parameters):
        """
        Check inputted parameters against the parameter and rate bounds.

        Parameters
        ----------
        parameters : list
            Inputted parameter to check.

        Returns
        -------
        paramsInsideBounds : bool
            True if the parameters are inside the bound. False if the parameters are outside of the bounds.

        """
        parameters = np.array(parameters)*self.bm.defaultParams
        
        # Check parameter boundaries
        if np.any(parameters <= self.lowerBounds) or np.any(parameters >= self.upperBounds):
            return False
        
        for comb in self.kCombinations:
            kLow = parameters[comb[0]] * np.exp(parameters[comb[1]] * self.vLow)
            kHigh = parameters[comb[0]] * np.exp(parameters[comb[1]] * self.vHigh)
            if comb[1] in [1,4]:
                if kHigh < self.km_min or kHigh > self.km_max:
                    return False
            else:
                if kLow < self.km_min or kLow > self.km_max:
                    return False
        
        # All tests passed!
        return True

def logTransforms(logTransforms, nParams):
    """
    Build a Pints transformation object to log transform some parameters.

    Parameters
    ----------
    logTransforms : list
        List of parameter indices to log transforms.
    nParams : int
        Number of parameters in the model.

    Returns
    -------
    transformation : Pints Transformation
        A pints transformation featuring log transforms on the indices specified by logTransforms and identify transforms (no transform) on the remaining paramters.

    """
    for i in range(nParams):
        if i == 0:
            if i in logTransforms:
                transformation = pints.LogTransformation(n_parameters=1)
            else:
                transformation = pints.IdentityTransformation(n_parameters=1)
        else:
            if i in logTransforms:
                transformation = pints.ComposedTransformation(transformation, pints.LogTransformation(n_parameters=1))
            else:
                transformation = pints.ComposedTransformation(transformation, pints.IdentityTransformation(n_parameters=1))
    
    return transformation

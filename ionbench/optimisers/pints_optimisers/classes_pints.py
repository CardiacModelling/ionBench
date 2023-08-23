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
    def __init__(self, bm, vHigh = 40, vLow = -120):
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
        self.bm = bm
        self.km_min = 1.67e-5
        self.km_max = 1e3
        self.vLow = vLow
        self.vHigh = vHigh
    
    def n_parameters(self):
        """
        Returns the number of parameters in the model.
        """
        return self.bm.n_parameters()
    
    def check(self, parameters):
        """
        Check inputted parameters against the parameter and rate bounds.

        Parameters
        ----------
        parameters : list
            Inputted parameter to check, in their original parameter space.

        Returns
        -------
        paramsInsideBounds : bool
            True if the parameters are inside the bound. False if the parameters are outside of the bounds.

        """
        parameters = self.bm.originalParameterSpace(parameters)
        
        # Check parameter boundaries
        if np.any(parameters <= self.bm.lb) or np.any(parameters >= self.bm.ub):
            return False
        
        #Check rate boundaries
        for rateTuple in self.bm._rateFunctions:
            rateFunc = rateTuple[0]
            rateType = rateTuple[1]
            if rateType == 'positive':
                #check kHigh is in bounds
                k = rateFunc(parameters, self.vHigh)
            elif rateType == 'negative':
                #check kLow is in bounds
                k = rateFunc(parameters, self.vLow)
            elif rateType == 'independent':
                #check rate in bounds
                k = rateFunc(parameters, 0) #Voltge doesn't matter
            else:
                print("Error in bm._rateFunctions. Doesn't contain 'positive', 'negative', or 'independent' in atleast one position. Check for typos.")
                k=0
            if k < self.km_min or k > self.km_max:
                return False
        
        # All tests passed!
        return True

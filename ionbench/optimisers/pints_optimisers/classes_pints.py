import pints
import numpy as np

class Model(pints.ForwardModel):
    """
    A Pints forwards model containing a benchmarker class.
    """
    def __init__(self, bm):
        """
        Initialise a Pints forward model with a benchmarker, linking up the n_parameters() and simulate() methods.

        Parameters
        ----------
        bm : benchmarker
            A test problem benchmarker.

        Returns
        -------
        None.

        """
        self.bm = bm
        
    def n_parameters(self):
        """
        Returns the number of parameters in the model

        Returns
        -------
        n_parameters : int
            Number of parameters in the model.

        """
        return self.bm.n_parameters()
    
    def simulate(self, parameters, times):
        """
        Simulates the model and returns the model output.

        Parameters
        ----------
        parameters : list
            A list of parameter values, length n_parameters().
        times : list
            A list of times at which to return model output.

        Returns
        -------
        out : list
            Model output, typically a current trace.

        """
        # Reset the simulation
        return self.bm.simulate(parameters, times)

class AdvancedBoundaries(pints.Boundaries):
    """
    Pints boundaries to apply to the parameters and the rates. 
    """
    def __init__(self, bm, vHigh = 40, vLow = -120):
        """
        Build a Pints boundary object to apply parameter and rate bounds.

        Parameters
        ----------
        bm : benchmarker
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
            Inputted parameter to check, in the input parameter space.

        Returns
        -------
        paramsInsideBounds : bool
            True if the parameters are inside the bound. False if the parameters are outside of the bounds.

        """
        parameters = self.bm.original_parameter_space(parameters)
        
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

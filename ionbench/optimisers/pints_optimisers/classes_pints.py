import pints
import numpy as np
from functools import lru_cache


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
        return self.sim(tuple(parameters))

    @lru_cache(maxsize=None)
    def sim(self, p):
        return self.bm.simulate(p, np.arange(0, self.bm.tmax, self.bm.freq))


class AdvancedBoundaries(pints.Boundaries):
    """
    Pints boundaries to apply to the parameters and the rates.
    """

    def __init__(self, bm):
        """
        Build a Pints boundary object to apply parameter and rate bounds.

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
            True if the parameters are inside the bound. False if the parameters are outside the bounds.

        """
        parameters = self.bm.original_parameter_space(parameters)
        return self.bm.in_rate_bounds(parameters) and self.bm.in_parameter_bounds(parameters)

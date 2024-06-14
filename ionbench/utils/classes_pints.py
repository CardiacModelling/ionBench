import pints
import numpy as np
import ionbench


def pints_setup(bm, x0, method, maxIter, debug, forceUnbounded=False):
    """
    Set up a Pints model and optimisation controller for a benchmarker.
    Parameters
    ----------
    bm : benchmarker
        A test problem benchmarker.
    x0 : list
        Initial parameter vector from which to start optimisation. If x0=None, a randomly sampled parameter vector is retrieved from bm.sample().
    method : pints.Optimiser
        A Pints optimiser to use. For example, pints.CMAES, pints.PSO, or pints.NelderMead.
    maxIter : int
        The maximum number of iterations to run for.
    debug : bool
        If debug is True, will increase the logging frequency.
    forceUnbounded : bool, optional
        If True, the optimisation will be forced to be unbounded. Default is False.

    Returns
    -------
    model : Model
        A Pints model containing the benchmarker.
    opt : pints.OptimisationController
        An optimisation controller to run the pints optimisation on.
    """
    if x0 is None:
        x0 = bm.sample()
    model = Model(bm)
    if 'moreno' in bm.NAME:  # pragma: no cover
        times = np.arange(len(bm.DATA))
    else:
        times = np.arange(0, bm.T_MAX, bm.TIMESTEP)
    problem = pints.SingleOutputProblem(model, times, model.bm.DATA)
    error = ErrorWithGrad(problem, bm)

    if bm.parametersBounded and not forceUnbounded:
        if bm.ratesBounded:
            boundaries = AdvancedBoundaries(bm)
        else:
            boundaries = pints.RectangularBoundaries(bm.input_parameter_space(bm.lb), bm.input_parameter_space(bm.ub))
        counter = 1
        while not boundaries.check(x0):  # pragma: no cover
            x0 = bm.sample()
            counter += 1
        if counter > 10:  # pragma: no cover
            print(f'Struggled to find parameters in bounds. Required {counter} iterations.')
        opt = pints.OptimisationController(error, x0, method=method, boundaries=boundaries)
    else:
        opt = pints.OptimisationController(error, x0, method=method)
    opt.set_max_iterations(maxIter)
    opt.set_max_unchanged_iterations(threshold=1e-7)
    if debug:
        opt.set_log_interval(iters=1)
    return model, opt


class ErrorWithGrad(pints.RootMeanSquaredError):
    """
    Pints error measure that uses RMSE and additionally defines the gradient of the RMSE.
    """
    def __init__(self, problem, bm):
        super().__init__(problem)
        self.bm = bm

    def evaluateS1(self, x):
        if 'moreno' in self.bm.NAME:  # pragma: no cover
            raise NotImplementedError('Moreno uses weighted RMSE in cost and grad, pints uses unweighted RMSE. No unweighted RMSE grad is available for Moreno.')
        return self.bm.grad(x, returnCost=True)


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
        super().__init__()

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
        return self.bm.signed_error(parameters)+self.bm.DATA


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

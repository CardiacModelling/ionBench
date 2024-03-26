import numpy as np


class Particle:
    """
    Default particle class for ionbench optimisers. Optimisers may use subclasses which override the set_position and set_velocity methods.
    Particles operate on the interval [0,1] which is then mapped to [lb,ub] for all optimisers to evaluate the cost.
    """
    def __init__(self, bm, cost_func, x0):
        if bm.parametersBounded:
            self.lb = bm.input_parameter_space(bm.lb)
            self.ub = bm.input_parameter_space(bm.ub)
        else:
            self.lb = None
            self.ub = None
        self.cost_func = cost_func
        self.position = np.zeros(len(x0))
        self.velocity = np.zeros(len(x0))
        self.currentCost = None
        self.set_position(x0)
        self.set_velocity()
        self.bestCost = np.inf  # Best cost of this particle
        self.bestPosition = np.copy(self.position)  # Position of best cost for this particle

    def set_position(self, x0):
        self.position = self.transform(x0 * np.random.uniform(low=0.9, high=1.1, size=len(x0)))
        self.clamp()

    def set_velocity(self):
        self.velocity = 0.1 * np.random.rand(len(self.position))

    def set_cost(self, cost=None):
        if cost is None:
            self.currentCost = self.cost_func(self.untransform(self.position))
        else:
            self.currentCost = cost
        if self.currentCost < self.bestCost:
            self.bestCost = self.currentCost
            self.bestPosition = np.copy(self.position)

    def transform(self, x):
        """
        Map from input space to [0,1] bounded space.

        Parameters
        ----------
        x : np.ndarray
            The parameters to transform.

        Returns
        -------
        xTrans : np.ndarray
            The transformed parameters.
        """
        xTrans = (x-self.lb) / (self.ub - self.lb)
        return xTrans

    def untransform(self, parameters):
        """
        Map from [0,1] to [lb,ub] in input parameter space.

        Parameters
        ----------
        parameters : np.ndarray
            The parameters to untransform.

        Returns
        -------
        xTrans : np.ndarray
            The untransformed parameters.
        """
        xTrans = parameters * (self.ub - self.lb) + self.lb
        return xTrans

    def clamp(self):
        """
        Clamp parameters to the input parameter space.
        """
        self.position = np.clip(self.position, 0, 1)

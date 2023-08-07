import pints
import numpy as np

class Model(pints.ForwardModel):
    def __init__(self, bm):
        self.bm = bm
        
    def n_parameters(self):
        return self.bm.n_parameters()
    
    def simulate(self, parameters, times):
        # Reset the simulation
        return self.bm.simulate(parameters, times)

class AdvancedBoundaries(pints.Boundaries):
    def __init__(self, paramCount, localBounds, kCombinations, bm, vHigh = 40, vLow = -120):
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
        return self.paramCount
    
    def check(self, parameters):
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

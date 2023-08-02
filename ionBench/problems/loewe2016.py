import ionBench
import myokit
import os
import numpy as np

class ikr(ionBench.benchmarker.Benchmarker):
    def __init__(self):
        print('Initialising Loewe 2016 IKr benchmark')
        self.model = myokit.load_model(os.path.join(ionBench.DATA_DIR, 'loewe2016', 'courtemanche-1998-ikr.mmt'))
        self._outputName = 'ikr.IKr'
        self._paramContainer = 'ikr'
        super().__init__()
        self.defaultParams = [3e-4, 14.1, 5, 3.3328, 5.1237, 1, 14.1, 6.5, 15, 22.4, 0.029411765, 138.994]
        self.additiveParams = [False, True, False, True, False, False, True, False, True, False, False, False]
        print('Benchmarker initialised')
    
    def sample(self, n=1):
        params = [None]*n
        for i in range(n):
            param = [None]*self.n_parameters()
            for j in range(self.n_parameters()):
                if self.additiveParams[j]:
                    param[j] = self.defaultParams[j] + np.random.uniform(-60,60)
                else:
                    param[j] = self.defaultParams[j]*10**np.random.uniform(-1,1) #Log uniform distribution
            params[i] = param
        if n==1:
            return params[0]
        else:
            return params
    
    def simulate(self, parameters, times):
        #Add parameter error to list
        self._paramRMSE.append(np.sqrt(np.mean((parameters-self.__trueParams)**2)))
        self._paramIdentifiedCount.append(np.sum(np.abs(parameters-self.__trueParams)<0.05))
        #Simulate the model and find the current
        # Reset the simulation
        self.sim.reset()
        
        if any(p<0 for p in parameters):
            return [np.inf]*len(times)
        
        # Update the parameters
        for i in range(len(self.defaultParams)):
            self.sim.set_constant(self._paramContainer+'.p'+str(i+1), parameters[i])
        
        # Run a simulation
        self.__solveCount += 1
        try:
            log = self.sim.run(self.tmax, log_times = times, log = [self._outputName])
            return log[self._outputName]
        except:
            return [np.inf]*len(times)

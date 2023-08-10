import numpy as np
import myokit
import csv
import os
import ionBench
import matplotlib.pyplot as plt
import warnings

class Tracker():
    def __init__(self):
        self.costs = []
        self.paramRMSRE = []
        self.paramIdentifiedCount = []
        self.solveCount = 0
    
    def update(self, trueParams, estimatedParams, cost = np.inf, incrementSolveCounter = True):
        trueParams = np.array(trueParams)
        estimatedParams = np.array(estimatedParams)
        self.paramRMSRE.append(np.sqrt(np.mean(((estimatedParams-trueParams)/trueParams)**2)))
        self.paramIdentifiedCount.append(np.sum(np.abs((estimatedParams-trueParams)/trueParams)<0.05))
        self.costs.append(cost)
        if incrementSolveCounter:
            self.solveCount += 1
    
    def plot(self):
        plt.figure()
        plt.scatter(range(len(self.costs)),self.costs, c="k", marker=".")
        plt.xlabel('Cost function calls')
        plt.ylabel('RMSE cost')
        plt.figure()
        plt.scatter(range(len(self.paramRMSRE)),self.paramRMSRE, c="k", marker=".")
        plt.xlabel('Cost function calls')
        plt.ylabel('Parameter RMSE')
        plt.figure()
        plt.scatter(range(len(self.paramIdentifiedCount)),self.paramIdentifiedCount, c="k", marker=".")
        plt.xlabel('Cost function calls')
        plt.ylabel('Number of parameters identified')
    
class Benchmarker():
    def __init__(self):
        self._bounded = False
        self._logTransformParams = [False]*self.n_parameters()
        self.plotter = True
        self.tracker = Tracker()
        self._updateTracker = True
        try:
            self.addModel(self.model, self._log)
        except:
            warnings.warn("No model or protocol has been defined for this benchmark. Please use the addModel function on this benchmarker object along with a Myokit model object and log to add the desired voltage protocol.")
    
    def addModel(self, model, log):
        self.model = model
        self.sim = myokit.Simulation(self.model)
        self.sim.set_tolerance(1e-8,1e-8)
        protocol = myokit.TimeSeriesProtocol(self._log.time(), self._log['voltage'])
        self.sim.set_protocol(protocol)
        self.tmax = self._log.time()[-1]
        self.sim.pre(500) #Prepace for 500ms
        
    def loadData(self, dataPath = '', paramPath = ''):
        if not dataPath == '':
            tmp=[]
            with open(dataPath, newline='') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    tmp.append(float(row[0]))
            self.data = np.array(tmp)
        if not paramPath == '':
            tmp=[]
            with open(paramPath, newline='') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    tmp.append(float(row[0]))
            self._trueParams = np.array(tmp)
    
    def addBounds(self, bounds):
        #Bounds checked against parameters in the inputted format, ie bounds on scaling factors, or bounds on log-transformed parameters, failing strict bounds does not increment cost counter
        self.lb = bounds[0]
        self.ub = bounds[1]
        self._bounded = True
    
    def logTransform(self, whichParams = []):
        if whichParams == []:
            whichParams = [True]*self.n_parameters()
        self._logTransformParams = whichParams
    
    def applyTransform(self, parameters):
        #Untransform any parameters
        for i in range(self.n_parameters()):
            if self._logTransformParams[i]:
                parameters[i] = np.exp(parameters[i])
        return parameters
    
    def inBounds(self, parameters):
        if self._bounded:
            if any(parameters[i]<self.lb[i] or parameters[i]>self.ub[i] for i in range(self.n_parameters())):
                return False
        return True
    
    def n_parameters(self):
        return len(self.defaultParams)
    
    def reset(self):
        self.sim.reset()
        self.tracker = Tracker()
    
    def cost(self, parameters, incrementSolveCounter = True):
        #Calculate cost for a given set of parameters
        testOutput = np.array(self.simulate(parameters, np.arange(0, self.tmax), incrementSolveCounter = incrementSolveCounter))
        return np.sqrt(np.mean((testOutput-self.data)**2))
    
    def signedError(self, parameters):
        #Calculate cost for a given set of parameters
        testOutput = np.array(self.simulate(parameters, np.arange(0, self.tmax)))
        return (testOutput-self.data)
    
    def squaredError(self, parameters):
        #Calculate cost for a given set of parameters
        testOutput = np.array(self.simulate(parameters, np.arange(0, self.tmax)))
        return (testOutput-self.data)**2
    
    def setParams(self, parameters):
        # Update the parameters
        for i in range(self.n_parameters()):
            if self._useScaleFactors:
                self.sim.set_constant(self._paramContainer+'.p'+str(i+1), self.defaultParams[i]*parameters[i])
            else:
                self.sim.set_constant(self._paramContainer+'.p'+str(i+1), parameters[i])
    
    def solveModel(self, parameters, times, continueOnError = True):
        if continueOnError:
            try:
                log = self.sim.run(self.tmax+1, log_times = times, log = [self._outputName])
                return log[self._outputName]
            except:
                return [np.inf]*len(times)
        else:
            log = self.sim.run(self.tmax+1, log_times = times, log = [self._outputName])
            return log[self._outputName]
    
    def simulate(self, parameters, times, continueOnError = True, incrementSolveCounter = True):
        parameters = self.applyTransform(parameters)
        #Simulate the model and find the current
        # Reset the simulation
        self.sim.reset()
        
        if not self.inBounds(parameters):
            self.tracker.update(self._trueParams, parameters, incrementSolveCounter = False)
            return [np.inf]*len(times)
        
        self.setParams(parameters)
        
        # Run a simulation
        out = self.solveModel(parameters, times, continueOnError = continueOnError)
        self.tracker.update(self._trueParams, parameters, cost = np.sqrt(np.mean((out-self.data)**2)), incrementSolveCounter = incrementSolveCounter)
        return out
    
    def evaluate(self, parameters):
        print('')
        print('=========================================')
        print('===    Evaluating Final Parameters    ===')
        print('=========================================')
        print('')
        print('Number of cost evaluations:      '+str(self.tracker.solveCount))
        cost =  self.cost(parameters, incrementSolveCounter = False)
        print('Final cost:                      {0:.6f}'.format(cost))
        print('Parameter RMSE:                  {0:.6f}'.format(self.tracker.paramRMSRE[-1]))
        print('Number of identified parameters: '+str(self.tracker.paramIdentifiedCount[-1]))
        print('Total number of parameters:      '+str(self.n_parameters()))
        print('')
        if self.plotter:
            self.tracker.plot()
        

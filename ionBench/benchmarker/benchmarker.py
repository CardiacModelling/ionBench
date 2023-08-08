import numpy as np
import myokit
import csv
import os
import ionBench
import matplotlib.pyplot as plt
import warnings
#TODO:
#Noise adds in bias, need to make sure it doesn't move optimal parameters [currently doesn't seem to be a big issue]
#Parallel processes won't give the correct solveCount. Will need to build a test case and see how I can resolve this. shared_memory from the multiprocessing class seems to be a good option

class Benchmarker():
    def __init__(self):
        self._solveCount = 0
        self._bounded = False
        self._logTransformParams = [False]*self.n_parameters()
        self.plotter = True
        self._costs = []
        self._paramRMSE = []
        self._paramIdentifiedCount = []
        try:
            self.addModel(self.model, self._log)
        except:
            warnings.warn("No model or protocol has been defined for this benchmark. Please use the addModel function on this object along with a Myokit model object and log to add the desired voltage protocol.")
    
    def addModel(self, model, log):
        self.model = model
        self.sim = myokit.Simulation(self.model)
        self.sim.set_tolerance(1e-6,1e-5)
        protocol = myokit.TimeSeriesProtocol(self._log.time(), self._log['voltage'])
        self.sim.set_protocol(protocol)
        self.tmax = self._log.time()[-1]
        self.sim.pre(500) #Prepace for 500ms
    
    def addBounds(self, bounds):
        #Bounds checked against parameters in the inputted format, ie bounds on scaling factors, or bounds on log-transformed parameters, failing strict bounds does not increment cost counter
        self.lb = bounds[0]
        self.ub = bounds[1]
        self._bounded = True
    
    def logTransform(self, whichParams = []):
        if whichParams == []:
            whichParams = [True]*self.n_parameters()
        self._logTransformParams = whichParams
        
    def n_parameters(self):
        return len(self.defaultParams)
    
    def reset(self):
        self._solveCount = 0
        self.sim.reset()
    
    def cost(self, parameters):
        #Calculate cost for a given set of parameters
        testOutput = np.array(self.simulate(parameters, np.arange(0, self.tmax)))
        tmp = np.sqrt(np.mean((testOutput-self.data)**2))
        self._costs.append(tmp)
        return tmp
    
    def signedError(self, parameters):
        #Calculate cost for a given set of parameters
        testOutput = np.array(self.simulate(parameters, np.arange(0, self.tmax)))
        self._costs.append(np.sqrt(np.mean((testOutput-self.data)**2)))
        return (testOutput-self.data)
    
    def squaredError(self, parameters):
        #Calculate cost for a given set of parameters
        testOutput = np.array(self.simulate(parameters, np.arange(0, self.tmax)))
        self._costs.append(np.sqrt(np.mean((testOutput-self.data)**2)))
        return (testOutput-self.data)**2
    
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
    
    def setParams(self, parameters):
        # Update the parameters
        for i in range(self.n_parameters()):
            if self._logTransformParams[i]:
                self.sim.set_constant(self._paramContainer+'.p'+str(i+1), self.defaultParams[i]*np.exp(parameters[i]))
            else:
                self.sim.set_constant(self._paramContainer+'.p'+str(i+1), self.defaultParams[i]*parameters[i])
            
    def simulate(self, parameters, times, continueOnError = True):
        #Add parameter error to list
        self._paramRMSE.append(np.sqrt(np.mean((parameters-self._trueParams)**2)))
        self._paramIdentifiedCount.append(np.sum(np.abs(parameters-self._trueParams)<0.05))
        #Simulate the model and find the current
        # Reset the simulation
        self.sim.reset()
        
        if self._bounded:
            if any(parameters[i]<self.lb[i] or parameters[i]>self.ub[i] for i in range(self.n_parameters())):
                return [np.inf]*len(times)
        self.setParams(parameters)
        
        # Run a simulation
        self._solveCount += 1
        if continueOnError:
            try:
                log = self.sim.run(self.tmax+1, log_times = times, log = [self._outputName])
                return log[self._outputName]
            except:
                return [np.inf]*len(times)
        else:
            log = self.sim.run(self.tmax+1, log_times = times, log = [self._outputName])
            return log[self._outputName]
    
    def evaluate(self, parameters):
        print('')
        print('=========================================')
        print('===    Evaluating Final Parameters    ===')
        print('=========================================')
        print('')
        solveCount = self._solveCount
        print('Number of cost evaluations:      '+str(self._solveCount))
        cost =  self.cost(parameters)
        print('Final cost:                      {0:.6f}'.format(cost))
        parameters = np.array(parameters)
        rmse = np.sqrt(np.mean((parameters-self._trueParams)**2))
        identifiedCount = np.sum(np.abs(parameters-self._trueParams)<0.05)
        print('Parameter RMSE:                  {0:.6f}'.format(rmse))
        print('Number of identified parameters: '+str(identifiedCount))
        print('Total number of parameters:      '+str(self.n_parameters()))
        print('')
        if self.plotter:
            plt.figure()
            plt.scatter(range(len(self._costs)),self._costs, c="k", marker=".")
            plt.xlabel('Cost function calls')
            plt.ylabel('RMSE cost')
            plt.figure()
            plt.scatter(range(len(self._paramRMSE)),self._paramRMSE, c="k", marker=".")
            plt.xlabel('Cost function calls')
            plt.ylabel('Parameter RMSE')
            plt.figure()
            plt.scatter(range(len(self._paramIdentifiedCount)),self._paramIdentifiedCount, c="k", marker=".")
            plt.xlabel('Cost function calls')
            plt.ylabel('Number of parameters identified')
        self._solveCount = solveCount

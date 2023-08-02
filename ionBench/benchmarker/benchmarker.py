import numpy as np
import myokit
import csv
import os
import ionBench
import matplotlib.pyplot as plt
#TODO:
#Prepacing - Still undecided if it is worth it in the benchmark
#Noise adds in bias, need to make sure it doesn't move optimal parameters [currently doesn't seem to be a big issue]
#Parallel processes won't give the correct solveCount. Will need to build a test case and see how I can resolve this. shared_memory from the multiprocessing class seems to be a good option
#Improve style of evaluate output
#fink2008 needs to have alpha rates moved out of exponential to match HH
class Benchmarker():
    def __init__(self):
        self.__solveCount = 0
        self.plotter = True
        self._costs = []
        self._paramRMSE = []
        self._paramIdentifiedCount = []
        self.sim = myokit.Simulation(self.model)
        self.sim.set_tolerance(1e-6,1e-5)
        log = myokit.DataLog.load_csv(os.path.join(ionBench.DATA_DIR, 'staircase-ramp.csv'))
        protocol = myokit.TimeSeriesProtocol(log.time(), log['voltage'])
        self.sim.set_protocol(protocol)
        self.tmax = log.time()[-1]
        
    def n_parameters(self):
        return len(self.defaultParams)
    
    def reset(self):
        self.__solveCount = 0
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
    
    def loadData(self, modelType):
        tmp=[]
        with open(os.path.join(ionBench.DATA_DIR, 'data'+modelType+'.csv'), newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                tmp.append(float(row[0]))
        self.data = np.array(tmp)
        tmp=[]
        with open(os.path.join(ionBench.DATA_DIR, 'trueParams'+modelType+'.csv'), newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                tmp.append(float(row[0]))
        self.__trueParams = np.array(tmp)
    
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
            self.sim.set_constant(self._paramContainer+'.p'+str(i+1), self.defaultParams[i]*parameters[i])
        
        # Run a simulation
        self.__solveCount += 1
        try:
            log = self.sim.run(self.tmax, log_times = times, log = [self._outputName])
            return log[self._outputName]
        except:
            return [np.inf]*len(times)
    
    def evaluate(self, parameters):
        print('Evaluating final parameters')
        print('Number of evaluations: '+str(self.__solveCount))
        cost =  self.cost(parameters)
        self.__solveCount -= 1
        print('Final cost: '+str(cost))
        parameters = np.array(parameters)
        rmse = np.sqrt(np.mean((parameters-self.__trueParams)**2))
        identifiedCount = np.sum(np.abs(parameters-self.__trueParams)<0.05)
        print('Parameter RMSE: '+str(rmse))
        print('Number of parameters correctly identified: '+str(identifiedCount))
        print('Total number of parameters in model: '+str(self.n_parameters()))
        print('Benchmark complete')
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

class HH_Benchmarker(Benchmarker):
    def __init__(self):
        print('Initialising Hodgkin-Huxley IKr benchmark')
        self.model = myokit.load_model(os.path.join(ionBench.DATA_DIR, 'beattie-2017-ikr-hh.mmt'))
        self._outputName = 'ikr.IKr'
        self._paramContainer = 'ikr'
        super().__init__()
        self.defaultParams = [2.26e-4, 0.0699, 3.45e-5, 0.05462, 0.0873, 8.91e-3, 5.15e-3, 0.03158, 0.1524]
        try:
            self.loadData('HH')
        except FileNotFoundError:
            self.data = None
            self.__trueParams = None
        print('Benchmarker initialised')

class MM_Benchmarker(Benchmarker):
    def __init__(self):
        print('Initialising Markov Model IKr benchmark')
        self.model = myokit.load_model(os.path.join(ionBench.DATA_DIR, "fink-2008-ikr-mm.mmt"))
        self._outputName = 'IKr.i_Kr'
        self._paramContainer = 'iKr_Markov'
        super().__init__()
        self.defaultParams = [-1.579, 0.0112, -3.168, -3.816, 0.0365, -0.872, 0.0223, -2.019, -0.0603, -2.514, -8.394, -0.0399, -3.182, -0.0312]
        try:
            self.loadData('MM')
        except FileNotFoundError:
            self.data = None
            self.__trueParams = None
        print('Benchmarker initialised')

def generateData(modelType):
    if modelType == 'HH':
        bm = HH_Benchmarker()
    elif modelType == 'MM':
        bm = MM_Benchmarker()
    trueParams = np.random.uniform(0.5,1.5,bm.n_parameters())
    out = bm.simulate(trueParams, np.arange(bm.tmax))
    out = out + np.random.normal(0, np.mean(np.abs(out))*0.05, len(out))
    with open('data'+modelType+'.csv', 'w', newline = '') as csvfile:
        writer = csv.writer(csvfile, delimiter = ',')
        writer.writerows(map(lambda x: [x], out))
    with open('trueParams'+modelType+'.csv', 'w', newline = '') as csvfile:
        writer = csv.writer(csvfile, delimiter = ',')
        writer.writerows(map(lambda x: [x], trueParams))

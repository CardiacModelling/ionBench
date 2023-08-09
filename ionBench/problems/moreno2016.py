import ionBench
import myokit
import os
import numpy as np
import csv

class ina(ionBench.benchmarker.Benchmarker):
    def __init__(self):
        print('Initialising Moreno 2016 INa benchmark')
        self.model = myokit.load_model(os.path.join(ionBench.DATA_DIR, 'moreno2016', 'moreno2016.mmt'))
        self._outputName = 'ina.INa'
        self._paramContainer = 'ina'
        self.defaultParams = np.array([7.6178e-3, 3.2764e1, 5.8871e-1, 1.5422e-1, 2.5898, 8.5072, 1.3760e-3, 2.888, 3.2459e-5, 9.5951, 1.3771, 2.1126e1, 1.1086e1, 4.3725e1, 4.1476e-2, 2.0802e-2])
        self._log = myokit.DataLog.load_csv(os.path.join(ionBench.DATA_DIR, 'moreno2016', 'protocol.csv'))
        self._trueParams = self.defaultParams
        self.loadData(dataPath = os.path.join(ionBench.DATA_DIR, 'moreno2016', 'ina.csv'))
        super().__init__()
        self.sim.set_tolerance(1e-14,1e-14)
        print('Benchmarker initialised')
    
    def sample(self, width=5):
        params = [None]*self.n_parameters()
        for j in range(self.n_parameters()):
            params[j] = self.defaultParams[j] * np.random.uniform(1-width/100,1+width/100)
        return params
    
    def setParams(self, parameters):
        # Update the parameters
        for i in range(self.n_parameters()):
            if self._logTransformParams[i]:
                self.sim.set_constant(self._paramContainer+'.p'+str(i+1), np.exp(parameters[i]))
            else:
                self.sim.set_constant(self._paramContainer+'.p'+str(i+1), parameters[i])
    
    def simulate(self, parameters, times, continueOnError = True):
        #Add parameter error to list
        for i in range(self.n_parameters()):
            if self._logTransformParams[i]:
                parameters[i] = np.exp(parameters[i])
        self._paramRMSE.append(np.sqrt(np.mean((parameters-self._trueParams)**2)))
        self._paramIdentifiedCount.append(np.sum(np.abs(parameters-self._trueParams)<0.05))
        #Simulate the model and find the current
        # Reset the simulation
        self.sim.reset()
        
        if self._bounded:
            if any(parameters[i]<self.lb[i] or parameters[i]>self.ub[i] for i in range(self.n_parameters())):
                return [np.inf]*len(times)
        self.setParams(parameters)
        
        measurementWindows = []
        #SSI/SSA measurements
        for i in range(9):
            measurementWindows.append([5000+i*10000, 5010+i*10000]) #Assumes peak falls in first 5ms
        #SSA/ACT measurement
        for i in range(20):
            measurementWindows.append([(i+10)*10000])
        #RFI measurements
        dt = [0.1, 0.5, 1, 3, 6, 9, 12, 15, 21, 30, 60, 90, 120, 210, 300, 450, 600, 900, 1500, 3000, 6000]
        for i in range(21):
            #Add first pulse
            measurementWindows.append([i*5000+295000,i*5000+295000+5]) #Assumes the peak falls in the first 5 ms
            #Add second pulse
            measurementWindows.append([i*5000+295100+dt[i],i*5000+295100+dt[i]+5]) #Assumes the peak falls in the first 5 ms
        #TAU measurements
        for i in range(9):
            measurementWindows.append([i*5000+410000,i*5000+410005]) #Assume peak and 50% of peak happen in first 5ms
        #RDUB measurements
        startTimes = [500000, 507630.48, 515000, 522779.98, 530000, 538377.98, 545000, 555171.98, 565000, 581450.98, 590000, 624390.98, 630000, 727180.98, 735000, 1011580.98, 1020000, 1924480.98, 1935000, 4633480.98]
        for i in startTimes:
            measurementWindows.append([i,i+25])
        logTimes = []
        windowRes = 100000
        for i in measurementWindows:
            if len(i)==2:
                logTimes += list(np.linspace(i[0],i[1],num=windowRes)) #100 points per windows
            else:
                #SSA/ACT protocol
                logTimes += i
        
        # Run a simulation
        self._solveCount += 1
        log = self.sim.run(self.tmax+1, log_times = logTimes, log = [self._outputName])
        #log = self.sim.run(self.tmax+1, log_times = logTimes)
        inaOut = log[self._outputName]
        inaOut = [-1*a for a in inaOut]
        ssi = [-1]*9
        print("")
        print("----ssi----")
        for i in range(len(ssi)):
            ssi[i] = max(inaOut[i*windowRes:(i+1)*windowRes])
            print(i)
            print(ssi[i])
        ssi = [a/max(ssi) for a in ssi]
        act = [-1]*20
        offset = len(ssi)*100
        print("")
        print("---act----")
        for i in range(len(act)):
            act[i] = inaOut[offset+i]
            print(i)
            print(act[i])
        act = [a/max(act) for a in act]
        rfi = [-1]*21
        offset = len(ssi)*windowRes+len(act)
        print("")
        print("----rfi----")
        for i in range(len(rfi)):
            firstPulse = max(inaOut[offset+i*200: offset+100+i*200])
            secondPulse = max(inaOut[offset+100+i*200: offset+200+i*200])
            rfi[i] = secondPulse/firstPulse
            print(i)
            print(dt[i])
            print(firstPulse)
            print(secondPulse)
            print(rfi[i])
        tau = [-1]*9
        offset = (len(ssi)+len(rfi)*2)*windowRes+len(act)
        print("")
        print("----tau----")
        for i in range(len(tau)):
            peak = max(inaOut[offset+i*100:offset+(i+1)*100])
            print(i)
            print(peak)
            reachedPeak = False
            for j in range(100):
                current = inaOut[offset+i*100+j]
                if current == peak:
                    reachedPeak = True
                    print("reached peak at j:"+str(j))
                if reachedPeak and abs(current)<=abs(peak/2):
                    tau[i] = logTimes[offset+i*100+j]-logTimes[offset+i*100]
                    print("Reached 50\%")
                    print(tau[i])
                    break
            print("Final current")
            print(current)
        rdub = [-1]*(len(startTimes)//2)
        offset = (len(ssi)+len(rfi)*2+len(tau))*windowRes+len(act)
        print("")
        print("----rdub----")
        for i in range(len(startTimes)//2):
            firstPulse = max(inaOut[offset+i*200:offset+i*200+100])
            lastPulse = max(inaOut[offset+i*200+100:offset+i*200+200])
            rdub[i] = lastPulse/firstPulse
            print(i)
            print(firstPulse)
            print(secondPulse)
            print(rdub[i])
        return ssi+act+rfi+tau+rdub
    #Cost function should be weighted by the number in each group

def generateData():
    bm = ina()
    out = bm.simulate(bm.defaultParams, np.arange(bm.tmax), continueOnError = False)
    with open(os.path.join(ionBench.DATA_DIR, 'moreno2016', 'ina.csv'), 'w', newline = '') as csvfile:
        writer = csv.writer(csvfile, delimiter = ',')
        writer.writerows(map(lambda x: [x], out))

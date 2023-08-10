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
        self._useScaleFactors = False
        self._log = myokit.DataLog.load_csv(os.path.join(ionBench.DATA_DIR, 'moreno2016', 'protocol.csv'))
        self._trueParams = self.defaultParams
        self.loadData(dataPath = os.path.join(ionBench.DATA_DIR, 'moreno2016', 'ina.csv'))
        super().__init__()
        print('Benchmarker initialised')
    
    def sample(self, width=5):
        params = [None]*self.n_parameters()
        for j in range(self.n_parameters()):
            params[j] = self.defaultParams[j] * np.random.uniform(1-width/100,1+width/100)
        return params
    
    def solveModel(self, parameters, times, continueOnError = True):
        if continueOnError:
            try:
                return self.runMoreno()
            except:
                return [np.inf]*59
        else:
            return self.runMoreno()
    
    def runMoreno(self):
        measurementWindows = []
        #SSI/SSA measurements
        for i in range(9):
            measurementWindows.append([5001+i*10000, (i+1)*10000])
        #SSA/ACT measurement
        for i in range(20):
            measurementWindows.append([(i+10)*10000])
        #RFI measurements
        dt = [0.1, 0.5, 1, 3, 6, 9, 12, 15, 21, 30, 60, 90, 120, 210, 300, 450, 600, 900, 1500, 3000, 6000]
        for i in range(21):
            #Add first pulse
            measurementWindows.append([i*5000+295001,i*5000+295001+99])
            #Add second pulse
            measurementWindows.append([i*5000+295100+dt[i]+0.01,i*5000+295100+dt[i]+0.01+25])
        #TAU measurements
        for i in range(9):
            measurementWindows.append([i*5000+405001,i*5000+406000])
        logTimes = []
        for i in measurementWindows:
            if len(i)==2:
                logTimes += list(np.linspace(i[0],i[1],num=100)) #100 points per windows
            else:
                #SSA/ACT protocol
                logTimes += i
        
        # Run a simulation
        log = self.sim.run(self.tmax+1, log_times = logTimes, log = [self._outputName])
        #log = self.sim.run(self.tmax+1, log_times = logTimes)
        inaOut = log[self._outputName]
        ssi = [None]*9
        for i in range(len(ssi)):
            ssi[i] = max(inaOut[i*100:(i+1)*100])
        ssi = [a/max(ssi) for a in ssi]
        act = [None]*20
        for i in range(len(act)):
            act[i] = inaOut[900+i]
        act = [a/max(act) for a in act]
        rfi = [None]*21
        for i in range(len(rfi)):
            firstPulse = max(inaOut[920+i*200:1020+i*200])
            secondPulse = max(inaOut[1020+i*200:1120+i*200])
            rfi[i] = secondPulse/firstPulse
        tau = [None]*9
        for i in range(len(tau)):
            peak = max(inaOut[5120+i*100:5120+(i+1)*100])
            reachedPeak = False
            for j in range(100):
                current = inaOut[5120+i*100+j]
                if current == peak:
                    reachedPeak = True
                if reachedPeak and abs(current)<=abs(peak/2):
                    tau[i] = logTimes[5120+i*100+j]-logTimes[5120+i*100]
                    break
        return ssi+act+rfi+tau

def generateData():
    bm = ina()
    out = bm.simulate(bm.defaultParams, np.arange(bm.tmax), continueOnError = False)
    with open(os.path.join(ionBench.DATA_DIR, 'moreno2016', 'ina.csv'), 'w', newline = '') as csvfile:
        writer = csv.writer(csvfile, delimiter = ',')
        writer.writerows(map(lambda x: [x], out))


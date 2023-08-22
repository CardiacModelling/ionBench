import ionBench
import myokit
import os
import numpy as np
import csv
import matplotlib.pyplot as plt

class ina(ionBench.benchmarker.Benchmarker):
    """
    The Moreno 2016 INa benchmarker. 
    
    The benchmarker uses the model from Moreno et al 2016 with a step protocol used to calculated summary curves which are then used for fitting. 
    
    Its parameters are specified as reported in Moreno et al 2016 with the true parameters being the same as the default and the center of the sampling distribution. 
    """
    def __init__(self):
        print('Initialising Moreno 2016 INa benchmark')
        self.model = myokit.load_model(os.path.join(ionBench.DATA_DIR, 'moreno2016', 'moreno2016.mmt'))
        self._outputName = 'ina.INa'
        self._paramContainer = 'ina'
        self.defaultParams = np.array([7.6178e-3, 3.2764e1, 5.8871e-1, 1.5422e-1, 2.5898, 8.5072, 1.3760e-3, 2.888, 3.2459e-5, 9.5951, 1.3771, 2.1126e1, 1.1086e1, 4.3725e1, 4.1476e-2, 2.0802e-2])
        self._rateFunctions = [(lambda p,V: 1/(p[0]*np.exp(-V/p[1])), 'negative'), (lambda p,V: p[2]/(p[0]*np.exp(-V/p[1])), 'negative'), (lambda p,V: p[3]/(p[0]*np.exp(-V/p[1])), 'negative'), (lambda p,V: 1/(p[4]*np.exp(V/p[5])), 'positive'), (lambda p,V: p[6]/(p[4]*np.exp(V/p[5])), 'positive'), (lambda p,V: p[7]/(p[4]*np.exp(V/p[5])), 'positive'), (lambda p,V: p[8]*np.exp(-V/p[9]), 'negative'), (lambda p,V: p[10]*np.exp(V/p[11]), 'positive'), (lambda p,V: p[12]*np.exp(V/p[13]), 'negative'), (lambda p,V: p[3]/(p[0]*np.exp(-V/p[1]))*p[12]*np.exp(V/p[13])*p[8]*np.exp(-V/p[9])/(p[7]/(p[4]*np.exp(V/p[5]))*p[10]*np.exp(V/p[11])), 'positive'), (lambda p,V: p[14]*p[12]*np.exp(V/p[13]), 'positive'), (lambda p,V: p[15]*p[8]*np.exp(-V/p[9]), 'negative')] #Used for rate bounds
        self._useScaleFactors = False
        self._log = myokit.DataLog.load_csv(os.path.join(ionBench.DATA_DIR, 'moreno2016', 'protocol.csv'))
        self._trueParams = self.defaultParams
        self.loadData(dataPath = os.path.join(ionBench.DATA_DIR, 'moreno2016', 'ina.csv'))
        super().__init__()
        #self.addProtocol()
        print('Benchmarker initialised')
    
    def addModel(self, model, log):
        self.model = model
        self.sim = myokit.Simulation(self.model)
        self.sim.set_tolerance(1e-8,1e-8)
        self._protocol = self.addProtocol()
        self.sim.set_protocol(self._protocol)
        self.tmax = self._protocol.characteristic_time()
        self.sim.pre(500) #Prepace for 500ms
    
    def sample(self, n=1, width=5):
        """
        Sample parameters for the Moreno 2016 problems. 

        Parameters
        ----------
        width : float, optional
            The width of the perturbation interval for sampling. The values used in Moerno et al 2016 are 5, 10, and 25. The default is 5.

        Returns
        -------
        params : list
            If n=1, then params is the vector of parameters. Otherwise, params is a list containing n parameter vectors.

        """
        params = [None]*n
        for i in range(n):
            param = [None]*self.n_parameters()
            for j in range(self.n_parameters()):
                param[j] = self.defaultParams[j] * np.random.uniform(1-width/100,1+width/100)
            params[i] = self.inputParameterSpace(param)
        if n==1:
            return params[0]
        else:
            return params
    
    def addProtocol(self):
        #Setup
        measurementWindows = []
        #windowSize = 1
        gap = 5000
        newProtocol = myokit.Protocol()
        
        #Protocol 1 - Measure peak current at -10mV after holding at voltages between -120mV and -40mV
        #Track start times
        protocolStartTimes = [0]
        #Create protocol
        protocol = myokit.pacing.steptrain_linear(vstart=-120, vend=-30, dv=10, vhold=-10, tpre=0, tstep=gap, tpost=gap)
        #Add windows to measure at
        for e in protocol.events():
            if e.level()==-10:
                measurementWindows.append([e.start(),e.stop(),'ssi'])
        #Add protocol to full protocol
        for e in protocol.events():
            newProtocol.schedule(level=e.level(), start=e.start(), duration=e.duration())
        #Add barrier to separate effects from different protocols
        newProtocol.schedule(level=-100, start=newProtocol.characteristic_time(), duration=gap)
        
        #Protocol 2 - Measure steady state current at varying voltages between -75mV and 20mV
        #Track start times
        protocolStartTimes.append(newProtocol.characteristic_time())
        #Create protocol
        protocol = myokit.pacing.steptrain_linear(vstart=-75, vend=25, dv=5, vhold=-100, tpre=gap, tstep=gap, tpost=0)
        #Add windows to measure at
        offset = newProtocol.characteristic_time()
        for e in protocol.events():
            if e.level()!=-100:
                measurementWindows.append([e.stop()+offset-0.01,'act']) #Just before transition
        #Add protocol to full protocol
        for e in protocol.events():
            newProtocol.schedule(level=e.level(), start=offset+e.start(), duration=e.duration())
        #Add barrier to separate effects from different protocols
        newProtocol.schedule(level=-100, start=newProtocol.characteristic_time(), duration=gap)
        
        #Protocol 3 - Ratio of max current at steps to -10mV with varying length gaps between
        #Track start times
        protocolStartTimes.append(newProtocol.characteristic_time())
        #Create protocol
        dt = [0.1, 0.5, 1, 3, 6, 9, 12, 15, 21, 30, 60, 90, 120, 210, 300, 450, 600, 900, 1500, 3000, 6000]
        vsteps = [-100, -10, -100, -10]*len(dt)+[-100]
        times = []
        for i in range(len(dt)):
            times.append(gap)
            times.append(100)
            times.append(dt[i])
            times.append(25)
        times.append(gap)
        protocol = myokit.Protocol()
        for i in range(len(times)):
            protocol.add_step(vsteps[i],times[i])
        #Add windows to measure at
        offset = newProtocol.characteristic_time()
        for e in protocol.events():
            if e.level()==-10:
                measurementWindows.append([e.start()+offset,e.stop()+offset,'rfi'])
        #Add protocol to full protocol
        for e in protocol.events():
            newProtocol.schedule(level=e.level(), start=offset+e.start(), duration=e.duration())
        #Add barrier to separate effects from different protocols
        newProtocol.schedule(level=-100, start=newProtocol.characteristic_time(), duration=gap)
        
        #Protocol 4 - Ratio of first and last max currents at 300 steps to -10mV with varying length gaps between
        #Track start times
        protocolStartTimes.append(newProtocol.characteristic_time())
        #Create protocol
        dt = [0.5, 1, 3, 9, 30, 90, 300, 900, 3000, 9000]
        vsteps = []
        times = []
        for i in range(len(dt)):
            vsteps.append(-100)
            times.append(gap)
            vsteps += [-10, -100]*300
            times += [25,dt[i]]*300
        times.append(gap)
        vsteps.append(-100)
        protocol = myokit.Protocol()
        for i in range(len(times)):
            protocol.add_step(vsteps[i],times[i])
        #Add windows to measure at
        offset = newProtocol.characteristic_time()
        tmpOffset = offset
        for i in range(len(dt)):
            measurementWindows.append([tmpOffset+gap,tmpOffset+gap+25,'rudb'])
            measurementWindows.append([tmpOffset+gap+(25+dt[i])*299,tmpOffset+gap+(25+dt[i])*299+25,'rudb'])
            tmpOffset = tmpOffset+gap+(25+dt[i])*300
        #Add protocol to full protocol
        for e in protocol.events():
            newProtocol.schedule(level=e.level(), start=offset+e.start(), duration=e.duration())
        #Add barrier to separate effects from different protocols
        newProtocol.schedule(level=-100, start=newProtocol.characteristic_time(), duration=gap)
        
        #Protocol 5 - Time to 50% of max current after step to between -20mV and 20mV
        #Track start times
        protocolStartTimes.append(newProtocol.characteristic_time())
        #Create protocol
        protocol = myokit.pacing.steptrain_linear(vstart=-20, vend=25, dv=5, vhold=-100, tpre=gap, tstep=gap, tpost=0)
        #Add windows to measure at
        offset = newProtocol.characteristic_time()
        for e in protocol.events():
            if e.level()!=-100:
                measurementWindows.append([e.start()+offset,e.stop()+offset,'tau'])
        #Add protocol to full protocol
        for e in protocol.events():
            newProtocol.schedule(level=e.level(), start=offset+e.start(), duration=e.duration())
        
        #Track total time
        protocolStartTimes.append(newProtocol.characteristic_time())
        
        #Store measurement windows
        self.sim.set_protocol(protocol)
        self.tmax = self._log.time()[-1]
        self.sim.pre(500) #Prepace for 500ms
        
        self._logTimes = []
        self._ssiBounds = []
        self._actBounds = []
        self._rfiBounds = []
        self._rudbBounds = []
        self._tauBounds = []
        
        for i in measurementWindows:
            if i[-1] == 'act':
                self._logTimes.append(i[0])
                self._actBounds.append(len(self._logTimes))
            else:
                lb = len(self._logTimes)
                self._logTimes += list(np.arange(i[0],i[1],0.005))
                ub = len(self._logTimes)
                if i[-1] == 'ssi':
                    self._ssiBounds.append([lb,ub])
                elif i[-1] == 'rfi':
                    self._rfiBounds.append([lb,ub])
                elif i[-1] == 'rudb':
                    self._rudbBounds.append([lb,ub])
                elif i[-1] == 'tau':
                    self._tauBounds.append([lb,ub])
        
        return newProtocol
    
    def solveModel(self, times, continueOnError = True):
        """
        Replaces the Benchmarker solveModel to call a special Moreno 2016 method (runMoreno()) which handles the summary curve calculations. The output is a vector of points on the summary curves.

        Parameters
        ----------
        times : list or numpy array
            Unneccessary for Moreno 2016. Only kept in since it will be passed in as an input by the main Benchmarker methods.
        continueOnError : bool, optional
            If continueOnError is True, any errors that occur during solving the model will be ignored and an infinite output will be given. The default is True.

        Returns
        -------
        modelOutput : list
            A vector of points on summary curves.

        """
        
        if continueOnError:
            try:
                return self.runMoreno()
            except:
                return [np.inf]*59
        else:
            return self.runMoreno()
    
    def runMoreno(self):
        """
        Runs the model to generate the Moreno et al 2016 summary curves. The points on these summary curves are then returned.

        Returns
        -------
        modelOutput : list
            A vector of points on summary curves.

        """
        
        # Run a simulation
        log = self.sim.run(self.tmax+1, log_times = self._logTimes, log = [self._outputName])
        inaOut = -np.array(log[self._outputName])
        
        ssi = [-1]*9
        for i in range(len(ssi)):
            ssi[i] = max(inaOut[self._ssiBounds[i][0]:self._ssiBounds[i][1]])
        ssi = [a/max(ssi) for a in ssi]
        
        act = [-1]*20
        for i in range(len(act)):
            act[i] = inaOut[self._actBounds[i]]
        act = [a/max(act) for a in act]
        
        rfi = [-1]*21
        for i in range(len(rfi)):
            firstPulse = max(inaOut[self._rfiBounds[2*i][0]:self._rfiBounds[2*i][1]])
            secondPulse = max(inaOut[self._rfiBounds[2*i+1][0]:self._rfiBounds[2*i+1][1]])
            rfi[i] = secondPulse/firstPulse
        
        rudb = [-1]*10
        for i in range(len(rudb)):
            firstPulse = max(inaOut[self._rudbBounds[2*i][0]:self._rudbBounds[2*i][1]])
            lastPulse = max(inaOut[self._rudbBounds[2*i+1][0]:self._rudbBounds[2*i+1][1]])
            rudb[i] = lastPulse/firstPulse
        
        tau = [-1]*9
        for i in range(len(tau)):
            peak = max(inaOut[self._tauBounds[i][0]:self._tauBounds[i][1]])
            reachedPeak = False
            for j in range(100):
                current = inaOut[self._tauBounds[i][0]+j]
                if current == peak:
                    reachedPeak = True
                    timeToPeak = self._logTimes[self._tauBounds[i][0]+j]
                if reachedPeak and abs(current)<=abs(peak/2):
                    tau[i] = self._logTimes[self._tauBounds[i][0]+j]-timeToPeak
                    break
        return [ssi,act,rfi,rudb,tau] #Weighted cost

def generateData():
    """
    Generate the data files for the Loewe 2016 benchmarker problems. The true parameters are the same as the deafult for these benchmark problems.

    Parameters
    ----------
    modelType : string
        'ikr' to generate the data for the IKr benchmark problem. 'ikur' to generate the data for the IKur benchmark problem.

    Returns
    -------
    None.

    """
    bm = ina()
    out = bm.simulate(bm.defaultParams, np.arange(bm.tmax), continueOnError = False)
    with open(os.path.join(ionBench.DATA_DIR, 'moreno2016', 'ina.csv'), 'w', newline = '') as csvfile:
        writer = csv.writer(csvfile, delimiter = ',')
        writer.writerows(map(lambda x: [x], out))

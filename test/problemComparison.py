import ionbench
import numpy as np

#Michael's approach
bmHH = ionbench.problems.staircase.HH_Benchmarker()
bmLoewe = ionbench.problems.loewe2016.ikr()

#Log transform applicable rates
bmHH.logTransform([True,False]*4+[False])
bmLoewe.logTransform([not i for i in bmLoewe.additiveParams])

#No scaling factors
bmHH._useScaleFactors = False
bmLoewe._useScaleFactors = False

#Bounds on rates and parameters - Needs implementing
lb = []
ub = []
for i in range(bmLoewe.n_parameters()):
    if bmLoewe.additiveParams[i]:
        lb.append(bmLoewe.defaultParams[i]-60*bmLoewe.paramSpaceWidth)
        ub.append(bmLoewe.defaultParams[i]+60*bmLoewe.paramSpaceWidth)
    else:
        lb.append(bmLoewe.defaultParams[i]*10**(-1*bmLoewe.paramSpaceWidth))
        ub.append(bmLoewe.defaultParams[i]*10**(1*bmLoewe.paramSpaceWidth))
bmLoewe.addBounds([lb,ub])

bmHH._useScaleFactors = False
lb = np.array(bmHH.defaultParams)*0.5
ub = np.array(bmHH.defaultParams)*1.5
bmHH.addBounds([lb,ub])

#CMA-ES run
for i in range(5):
    bmHH.reset()
    ionbench.optimisers.pints_optimisers.cmaes_pints.run(bmHH)

for i in range(5):
    bmLoewe.reset()
    ionbench.optimisers.pints_optimisers.cmaes_pints.run(bmLoewe, maxIter=3000)


#Loewe et al 2016 approach
bmHH = ionbench.problems.staircase.HH_Benchmarker()
bmLoewe = ionbench.problems.loewe2016.ikr()

#Bounds on parameters
lb = []
ub = []
for i in range(bmLoewe.n_parameters()):
    if bmLoewe.additiveParams[i]:
        lb.append(bmLoewe.defaultParams[i]-60*bmLoewe.paramSpaceWidth)
        ub.append(bmLoewe.defaultParams[i]+60*bmLoewe.paramSpaceWidth)
    else:
        lb.append(bmLoewe.defaultParams[i]*10**(-1*bmLoewe.paramSpaceWidth))
        ub.append(bmLoewe.defaultParams[i]*10**(1*bmLoewe.paramSpaceWidth))
bmLoewe.addBounds([lb,ub])

bmHH._useScaleFactors = False
lb = np.array(bmHH.defaultParams)*0.5
ub = np.array(bmHH.defaultParams)*1.5
bmHH.addBounds([lb,ub])

#Hybrid PSO TRR run
ionbench.optimisers.external_optimisers.hybridPSOTRR_Loewe2016.run(bmHH, debug = True)
ionbench.optimisers.external_optimisers.hybridPSOTRR_Loewe2016.run(bmLoewe,debug = True)

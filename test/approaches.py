import ionbench
import numpy as np

#Load approaches with all settings
empty = ionbench.approach.Approach()
full = ionbench.approach.Approach(logTransform='Full',bounds='positive',scaleFactors='on')
problemSpecific = ionbench.approach.Approach(logTransform='standard',bounds='Sampler',scaleFactors='On')

bmHH = ionbench.problems.staircase.HH_Benchmarker()
bmMM = ionbench.problems.staircase.MM_Benchmarker()
bmLoeweIKr = ionbench.problems.loewe2016.ikr()
bmLoeweIKur = ionbench.problems.loewe2016.ikur()
bmMorenoINa = ionbench.problems.moreno2016.ina()
#%%
bm = [bmHH, bmMM, bmLoeweIKr, bmLoeweIKur, bmMorenoINa]
for i in range(len(bm)):
    empty.apply(bm[i])
    print(not bm[i]._bounded)
    print(not any(bm[i]._logTransformParams))
    print(not bm[i]._useScaleFactors)
    bm[i].cost(bm[i].sample())

for i in range(len(bm)):
    problemSpecific.apply(bm[i])
    print(bm[i]._bounded)
    print(any(bm[i]._logTransformParams) and not all(bm[i]._logTransformParams))
    print(bm[i]._useScaleFactors)
    bm[i].cost(bm[i].sample())

for i in range(len(bm)):
    full.apply(bm[i])
    print(bm[i]._bounded)
    print(all(bm[i]._logTransformParams))
    print(bm[i]._useScaleFactors)
    bm[i].cost(bm[i].sample())


#Custom approach
bmHH = ionbench.problems.staircase.HH_Benchmarker()
custom = ionbench.approach.Approach(logTransform='custom', bounds='custom', scaleFactors='off', customLogTransform = [True, True, False]*3, customBounds = [[-1]*3+[0]*3+[-np.inf]*3, [1]*3+[100]*3+[np.inf]*3])
custom.apply(bmHH)
print(bmHH._bounded)
print(any(bmHH._logTransformParams) and not all(bmHH._logTransformParams))
print(not bmHH._useScaleFactors)
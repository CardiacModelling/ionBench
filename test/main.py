import ionBench
import numpy as np

bmHH = ionBench.problems.staircase.HH_Benchmarker()
bmMM = ionBench.problems.staircase.MM_Benchmarker()
bmLoeweIKr = ionBench.problems.loewe2016.ikr()
bmLoeweIKur = ionBench.problems.loewe2016.ikur()
bmMorenoINa = ionBench.problems.moreno2016.ina()

#%% Check benchmarkers
bm = [bmHH, bmMM, bmLoeweIKr, bmLoeweIKur, bmMorenoINa]
problemNames = ['HH', 'MM', 'Loewe 2016 IKr', 'Loewe 2016 IKur', 'Moreno 2016 INa']
for i in range(len(bm)):
    bm[i].plotter = False
    print('===================')
    print(problemNames[i])
    print('Check evaluate increments exactly once when called default rates')
    try:
        bm[i].evaluate(bm[i].defaultParams)
        bm[i].simulate(bm[i].defaultParams, times = np.arange(bm[i].tmax), continueOnError = False)
        bm[i].evaluate(bm[i].defaultParams)
    except Exception as e:
        print(e)

for i in range(len(bm)):
    print('===================')
    print(problemNames[i])
    print('Check bounds don\'t cause errors')
    print('There should be an infinite cost for MM only')
    try:
        b = bm[i]
        bounds = [[0]*b.n_parameters(),[np.inf]*b.n_parameters()]
        b.addBounds(bounds)
        b.evaluate(b.defaultParams)
    except Exception as e:
        print(e)
    
for i in range(len(bm)):
    print('===================')
    print(problemNames[i])
    print('Check log transforms don\'t cause errors')
    try:
        bm[i]._bounded=False
        b = bm[i]
        b.logTransform(whichParams = [True]*b.n_parameters())
        if i<2:#HH and MM
            b.evaluate([0]*b.n_parameters())
        else:
            b.evaluate(np.log(b.defaultParams))
    except Exception as e:
        print(e)

#%% Scipy algorithms
print("========================")
print("++++++++++++++++++++++++")
print("========================")
print("lm_scipy")
for i in range(len(bm)):
    print("-------------------")
    print(problemNames[i])
    try:
        ionBench.optimisers.scipy_optimisers.lm_scipy.run(bm[i],bm[i].defaultParams, maxfev = 100)
    except Exception as e:
        print(e)

print("++++++++++++++++++++++++")
print("nelderMead_scipy")
for i in range(len(bm)):
    print("-------------------")
    print(problemNames[i])
    try:
        ionBench.optimisers.scipy_optimisers.nelderMead_scipy.run(bm[i],bm[i].defaultParams, maxfev = 100)
    except Exception as e:
        print(e)

print("++++++++++++++++++++++++")
print("powell_scipy")
for i in range(len(bm)):
    print("-------------------")
    print(problemNames[i])
    try:
        ionBench.optimisers.scipy_optimisers.powell_scipy.run(bm[i],bm[i].defaultParams, maxfev = 100)
    except Exception as e:
        print(e)

print("++++++++++++++++++++++++")
print("trf_scipy")
for i in range(len(bm)):
    print("-------------------")
    print(problemNames[i])
    try:
        ionBench.optimisers.scipy_optimisers.trustRegionReflective_scipy.run(bm[i],bm[i].defaultParams, maxfev = 100)
    except Exception as e:
        print(e)

#%% Pints algorithms
print("========================")
print("++++++++++++++++++++++++")
print("========================")
print("cmaes-pints")
kCombinations = [[0,1], [2,3], [4,5], [6,7]]
logTransforms = [0, 2, 4, 6]
localBounds = [[0,1e-7,1e3], [1,1e-7,0.4], [2,1e-7,1e3], [3,1e-7,0.4], [4,1e-7,1e3], [5,1e-7,0.4], [6,1e-7,1e3], [7,1e-7,0.4]]
iterCount = 2
maxIter = 50
try:
    ionBench.optimisers.pints_optimisers.cmaes_pints.run(bm, kCombinations, localBounds = localBounds, logTransforms = logTransforms, iterCount=iterCount, maxIter=maxIter)
except Exception as e:
    print(e)

print("++++++++++++++++++++++++")
print("nelderMead-pints")
try:
    ionBench.optimisers.pints_optimisers.nelderMead_pints.run(bm, logTransforms = logTransforms, iterCount=iterCount, maxIter=maxIter)
except Exception as e:
    print(e)

print("++++++++++++++++++++++++")
print("pso-pints")
try:
    ionBench.optimisers.pints_optimisers.pso_pints.run(bm, logTransforms = logTransforms, iterCount=iterCount, maxIter=maxIter)
except Exception as e:
    print(e)

print("++++++++++++++++++++++++")
print("snes-pints")
try:
    ionBench.optimisers.pints_optimisers.snes_pints.run(bm, logTransforms = logTransforms, iterCount=iterCount, maxIter=maxIter)
except Exception as e:
    print(e)
    
print("++++++++++++++++++++++++")
print("xnes-pints")
try:
    ionBench.optimisers.pints_optimisers.xnes_pints.run(bm, logTransforms = logTransforms, iterCount=iterCount, maxIter=maxIter)
except Exception as e:
    print(e)

#%% External optimisers
print("========================")
print("++++++++++++++++++++++++")
print("========================")
print("GA Bot 2012")
for i in range(len(bm)):
    print("-------------------")
    print(problemNames[i])
    try:
        ionBench.optimisers.external_optimisers.GA_Bot2012.run(bm[i], nGens = 5, popSize = 10, debug = True)
    except Exception as e:
        print(e)

print("++++++++++++++++++++++++")
print("GA Smirnov 2020")
for i in range(len(bm)):
    print("-------------------")
    print(problemNames[i])
    try:
        ionBench.optimisers.external_optimisers.GA_Bot2012.run(bm[i], nGens = 5, popSize = 10, debug = True)
    except Exception as e:
        print(e)

print("++++++++++++++++++++++++")
print("PatternSearch Kohjitani 2020")
for i in range(len(bm)):
    print("-------------------")
    print(problemNames[i])
    try:
        ionBench.optimisers.external_optimisers.patternSearch_Kohjitani2022.run(bm[i], maxfev = 100, debug = True)
    except Exception as e:
        print(e)

print("++++++++++++++++++++++++")
print("PPSO Chen 2012")
groups = [[0,2,4,6],[1,3,5,7],[8]]
for i in range(len(bm)):
    print("-------------------")
    print(problemNames[i])
    try:
        ionBench.optimisers.external_optimisers.ppso_chen2012.run(bm[i], groups= groups, debug = True)
    except Exception as e:
        print(e)

print("++++++++++++++++++++++++")
print("SPSA")
for i in range(len(bm)):
    print("-------------------")
    print(problemNames[i])
    try:
        ionBench.optimisers.spsa_spsa.run(bm[i], x0=bm[i].defaultParams,maxiter=50)
    except Exception as e:
        print(e)


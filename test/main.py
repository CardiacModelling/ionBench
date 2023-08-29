import ionbench
import numpy as np

bmHH = ionbench.problems.staircase.HH_Benchmarker()
bmMM = ionbench.problems.staircase.MM_Benchmarker()
bmLoeweIKr = ionbench.problems.loewe2016.ikr()
bmLoeweIKur = ionbench.problems.loewe2016.ikur()
bmMorenoINa = ionbench.problems.moreno2016.ina()

#%% Check benchmarkers
bm = [bmHH, bmMM, bmLoeweIKr, bmLoeweIKur, bmMorenoINa]
problemNames = ['HH', 'MM', 'Loewe 2016 IKr', 'Loewe 2016 IKur', 'Moreno 2016 INa']
for i in range(len(bm)):
    bm[i].plotter = False
    print('===================')
    print(problemNames[i])
    print('Check evaluate increments exactly once when called')
    try:
        params = bm[i].sample()
        bm[i].evaluate(params)
        bm[i].simulate(params, times = np.arange(bm[i].tmax), continueOnError = False)
        bm[i].evaluate(params)
    except Exception as e:
        print(e)

for i in range(len(bm)):
    print('===================')
    print(problemNames[i])
    print('Check bounds don\'t cause errors')
    try:
        b = bm[i]
        bounds = [[0]*b.n_parameters(),[np.inf]*b.n_parameters()]
        b.add_bounds(bounds)
        b.evaluate(b.sample())
    except Exception as e:
        print(e)
    
for i in range(len(bm)):
    print('===================')
    print(problemNames[i])
    print('Check log transforms don\'t cause errors')
    try:
        bm[i]._bounded=False
        b = bm[i]
        if "loewe" in b._name:
            b.log_transform(whichParams = b.standardLogTransform)
        else:
            b.log_transform()
        b.evaluate(b.sample())
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
        ionbench.optimisers.scipy_optimisers.lm_scipy.run(bm[i],bm[i].sample(), maxfev = 100)
    except Exception as e:
        print(e)

print("++++++++++++++++++++++++")
print("nelderMead_scipy")
for i in range(len(bm)):
    print("-------------------")
    print(problemNames[i])
    try:
        ionbench.optimisers.scipy_optimisers.nelderMead_scipy.run(bm[i],bm[i].sample(), maxfev = 100)
    except Exception as e:
        print(e)

print("++++++++++++++++++++++++")
print("powell_scipy")
for i in range(len(bm)):
    print("-------------------")
    print(problemNames[i])
    try:
        ionbench.optimisers.scipy_optimisers.powell_scipy.run(bm[i],bm[i].sample(), maxfev = 100)
    except Exception as e:
        print(e)

print("++++++++++++++++++++++++")
print("trf_scipy")
for i in range(len(bm)):
    print("-------------------")
    print(problemNames[i])
    try:
        ionbench.optimisers.scipy_optimisers.trustRegionReflective_scipy.run(bm[i],bm[i].sample(), maxfev = 100)
    except Exception as e:
        print(e)

#%% Pints algorithms
print("========================")
print("++++++++++++++++++++++++")
print("========================")
print("cmaes-pints")
maxIter = 50
try:
    ionbench.optimisers.pints_optimisers.cmaes_pints.run(bm[0], maxIter=maxIter)
except Exception as e:
    print(e)

print("++++++++++++++++++++++++")
print("nelderMead-pints")
try:
    ionbench.optimisers.pints_optimisers.nelderMead_pints.run(bm[0], maxIter=maxIter)
except Exception as e:
    print(e)

print("++++++++++++++++++++++++")
print("pso-pints")
try:
    ionbench.optimisers.pints_optimisers.pso_pints.run(bm[0], maxIter=maxIter)
except Exception as e:
    print(e)

print("++++++++++++++++++++++++")
print("snes-pints")
try:
    ionbench.optimisers.pints_optimisers.snes_pints.run(bm[0], maxIter=maxIter)
except Exception as e:
    print(e)
    
print("++++++++++++++++++++++++")
print("xnes-pints")
try:
    ionbench.optimisers.pints_optimisers.xnes_pints.run(bm[0], maxIter=maxIter)
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
        ionbench.optimisers.external_optimisers.GA_Bot2012.run(bm[i], nGens = 5, popSize = 10, debug = True)
    except Exception as e:
        print(e)

print("++++++++++++++++++++++++")
print("GA Smirnov 2020")
for i in range(len(bm)):
    print("-------------------")
    print(problemNames[i])
    try:
        ionbench.optimisers.external_optimisers.GA_Bot2012.run(bm[i], nGens = 5, popSize = 10, debug = True)
    except Exception as e:
        print(e)

print("++++++++++++++++++++++++")
print("PatternSearch Kohjitani 2020")
for i in range(len(bm)):
    print("-------------------")
    print(problemNames[i])
    try:
        ionbench.optimisers.external_optimisers.patternSearch_Kohjitani2022.run(bm[i], maxfev = 100, debug = True)
    except Exception as e:
        print(e)

print("++++++++++++++++++++++++")
print("PPSO Chen 2012")
groups = [[0,2,4,6],[1,3,5,7],[8]]
for i in range(len(bm)):
    print("-------------------")
    print(problemNames[i])
    try:
        ionbench.optimisers.external_optimisers.ppso_Chen2012.run(bm[i], groups= groups, debug = True)
    except Exception as e:
        print(e)

print("++++++++++++++++++++++++")
print("SPSA")
for i in range(len(bm)):
    print("-------------------")
    print(problemNames[i])
    try:
        ionbench.optimisers.spsa_spsa.run(bm[i], x0=bm[i].sample(),maxiter=50)
    except Exception as e:
        print(e)

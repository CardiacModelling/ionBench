import numpy as np
import benchmarker
import spsa
import time
#To do:
#Convergence needs to massively improve. An implementation of the original Spall 1992 algorithm might actually help
try: bm
except NameError:
    print('No benchmarker loaded. Creating a new one')
    bm = benchmarker.HH_Benchmarker()
else: bm.reset()

x0 = np.ones(bm.n_parameters())

bm.reset()
printFrequency = 100
counter = 0
for variables in spsa.iterator.minimize(spsa.with_input_noise(bm.cost, shape=bm.n_parameters(), noise = 0.01), x0, adam=False):
    if counter % printFrequency == 0:
        print('-----------')
        print('Current cost:')
        print(variables['y'])
        print('Found at point:')
        print(variables['x'])
        print('Best cost:')
        print(variables['y_best'])
        print('Found at point:')
        print(variables['x_best'])
        time.sleep(0.1)
    counter += 1
out = variables['x_best']

#out = spsa.minimize(bm.cost, x0, iterations=20)
#bm.evaluate(out)

#bm.reset()

#out2 = spsa.minimize(spsa.with_input_noise(bm.cost, shape=bm.n_parameters(), noise = 0.5), x0, iterations = 5)
#bm.evaluate(out)

from ionBench.problems import staircase
import spsa
#To do:
#Convergence needs to massively improve. An implementation of the original Spall 1992 algorithm might actually help

def run(bm, x0, printFrequency=100, maxiter = 1000):
    """
    Runs SPSA (Simultaneous Perturbation Stochastic Approximation) from Pints using a benchmarker. 

    Parameters
    ----------
    bm : Benchmarker
        A benchmarker to evaluate the performance of the optimisation algorithm.
    x0 : list
        Initial parameter vector from which to start optimisation.
    printFrequency : int, optional
        Number of iterations to wait between printing output
    maxIter : int, optional
        Number of iterations of SPSA to run per repeat. The default is 1000.

    Returns
    -------
    xbest : list
        The best parameters identified by SPSA.

    """
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
        counter += 1
        if counter > maxiter:
            break
    
    bm.evaluate(variables['x_best'])
    return variables['x_best']

if __name__ == '__main__':
    bm = staircase.HH_Benchmarker()
    x0 = bm.defaultParams
    run(bm, x0)
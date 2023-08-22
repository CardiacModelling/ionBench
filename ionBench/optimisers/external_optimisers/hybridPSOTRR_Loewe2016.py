import numpy as np
import scipy
from ionBench.problems import staircase

def run(bm, n=96, K=5, L=250, phi1=2.05, phi2=2.05, debug=False):
    """
    Runs the hybrid particle swarm optimisation - trust region reflective algorithm from Loewe et al 2016. If the benchmarker is bounded, the solver will search in the interval [lb,ub], otherwise the solver will search in the interval [0,2*default].
    
    Notes on using this optimiser:
        It is unclear the specifics of the optimisation used in Loewe et al 2016. It does not describe the method for sampling the initial positions or velocities of the particles (we use uniform distribution between bounds for position and expect Loewe et al 2016 did similarly, and zeros for the initial velocities, waiting for them to be set by TRR). The initial velocities may be different to those used in the Loewe et al 2016, but luckily the algorithm quickly overwrites the velocities after the first iteration based on the movement under TRR so it is only a difference between randomly sampled points and running TRR (our method) or randomly sampling points, moving them in a random direction and applying the bounds then applying TRR (a method with a non-zero random velocity).
        The TRR method in Loewe et al 2016 using Matlab's lsqnonlin implementation. As a Python based substitute we have used Scipy. Unfortunately, the Scipy implementation does not allow a maximum number of iterations to be specified. We have specified the maximum number of cost function evaluations (which appears to not include calls to calculate the gradient), which is strictly more than the number of iterations, but still of a comparable size. There may also be differences in how the radius of the trust region is calculated and updated between these two implementations. 
        
    Parameters
    ----------
    bm : Benchmarker
        A benchmarker to evaluate the performance of the optimisation algorithm.
    n : int, optional
        Number of particles. The default is 96.
    K : int, optional
        Number of function calls allowed in TRR. Note that this uses the function calls reported by scipy, similar in scale to the number of iterations of TRR used in Loewe et al 2016. This is not the true number of function calls to the benchmarker as this appears to not include finite difference gradient calculations. The default is 5.
    lmax : int, optional
        Maximum number of iterations. The default is 250.
    phi1 : float, optional
        Scale of the acceleration towards a particles best positions. The default is 2.05.
    phi2 : float, optional
        Scale of the acceleration towards the best point seen across all particles. The default is 2.05.
    debug : bool, optional
        If True, debug information will be printed, reporting that status of the optimisation each generation. The default is False.

    Returns
    -------
    xbest : list
        The best parameters identified.

    """
    
    class particle:
        def __init__(self, n_param):
            self.velocity = np.zeros(bm.n_parameters())
            self.position = np.random.rand(n_param)
            self.bestCost = np.inf #Best cost of this particle
            self.bestPosition = np.copy(self.position) #Position of best cost for this particle
            self.currentCost = None
        
        def setCost(self, cost):
            self.currentCost = cost
            if cost < self.bestCost:
                self.bestCost = cost
                self.bestPosition = np.copy(self.position)
    
    def costFunc(x):
        return bm.cost(transform(x))
    
    def signedError(x):
        return bm.signedError(transform(x))
    
    def evaluate(x):
        return bm.evaluate(transform(x))
    
    def transform(x):
        if bm._bounded:
            xTrans = bm.lb + x*(bm.ub-bm.lb) #Map x to [lb,ub]
        else:
            xTrans = x*2 #Map x from [0,1] to [0,2]
            if not bm._useScaleFactors:
                xTrans = xTrans*bm.defaultParams #Map to [0,2*default]
        return xTrans
    
    if (phi1 + phi2)**2-4*(phi1 + phi2)<0:
        print("Invalid constriction factor using specified values for phi1 and phi2. Using defaults of phi1=phi2=2.05 instead.")
        phi1 = 2.05
        phi2 = 2.05
    phi = phi1 + phi2
    constFactor = 2/(phi-2+np.sqrt(phi**2-4*phi))
    
    if debug:
        verbose = 1
    else:
        verbose = 0
    
    particleList = []
    for i in range(n):
        particleList.append(particle(bm.n_parameters()))
    
    Gcost = [np.inf]*L #Best cost ever
    Gpos = [None]*L #Position of best cost ever
    for l in range(L):
        if l > 0:
            Gcost[l] = Gcost[l-1]
            Gpos[l] = Gpos[l-1]
        
        if debug:
            print('-------------')
            print("Begginning population: "+str(l))
            print("Best cost so far: "+str(Gcost[l]))
            print("Found at position: "+str(Gpos[l]))
        
        #Find best positions, both globally and locally
        for p in particleList:
            cost = costFunc(p.position)
            p.setCost(cost)
            if cost < Gcost[l]:
                Gcost[l] = cost
                Gpos[l] = np.copy(p.position)
        
        #Update velocities
        for p in particleList:
            localAcc = phi1*np.random.rand()*(p.bestPosition-p.position)
            globalAcc = phi2*np.random.rand()*(Gpos[l]-p.position)
            p.velocity = constFactor*(p.velocity + localAcc + globalAcc)
        if debug:
            print("Velocities renewed")
        
        #Enforce bounds
        for p in particleList:
            p.position += p.velocity
            for i in range(bm.n_parameters()):
                if p.position[i] < 0:
                    p.position[i] = np.random.rand()/4
                elif p.position[i] > 1:
                    p.position[i] = 1-np.random.rand()/4
        
        #Iterations of TRR
        if debug:
            print("Begginning TRR")
        bounds = ([0]*bm.n_parameters(),[1]*bm.n_parameters())
        for p in particleList:
            out = scipy.optimize.least_squares(signedError, p.position, method='trf', diff_step=1e-3, max_nfev = 2*K, bounds = bounds, verbose=verbose)
            p.velocity = out.x - p.position
            p.position = out.x
            
        if debug:
            print("Positions renewed")
            print("Finished population: "+str(l))
            print("Best cost so far: "+str(Gcost[l]))
            print("Found at position: "+str(Gpos[l]))

    evaluate(Gpos[l])
    return Gpos[l]

if __name__ == '__main__':
    bm = staircase.HH_Benchmarker()
    run(bm,debug=True)
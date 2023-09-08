import numpy as np
import ionbench
import itertools
#Notes: The algorithm defines parameters between 0 and 1, this is mapped to 0 to 2 when the cost function is called

def run(bm, groups = [], n=20, c1=1.4, c2=1.4, qmax=5, lmax=200, gmin=0.05, w=0.6, debug=False):
    """
    Runs the perturbed particle swarm optimisation algorithm from Chen et al 2012. If the benchmarker is bounded, the solver will search in the interval [lb,ub], otherwise the solver will search in the interval [0,2*default]

    Parameters
    ----------
    bm : Benchmarker
        A benchmarker to evaluate the performance of the optimisation algorithm.
    groups : list, optional
        Groupings of parameters to use in the algorithm. Default is [], in which case every parameter will be in its own group.
    n : int, optional
        Number of particles. The default is 20.
    c1 : float, optional
        Scale of the acceleration towards a particles best positions. The default is 1.4.
    c2 : float, optional
        Scale of the acceleration towards the best point seen across all particles. The default is 1.4.
    qmax : int, optional
        Maximum number of iterations without improvement before perturbations. The default is 5.
    lmax : int, optional
        Maximum number of iterations. The default is 200.
    gmin : float, optional
        Target cost. Once the cost is reduced below gmin, the optimisation terminates. The default is 0.05.
    w : float, optional
        Initial value for the inertia of the particles. The default is 0.6.
    debug : bool, optional
        If True, debug information will be printed, reporting that status of the optimisation each generation. The default is False.

    Returns
    -------
    xbest : list
        The best parameters identified.

    """
    
    class particle:
        def __init__(self, n_param):
            self.velocity = 0.1*np.random.rand(n_param)
            self.position = np.random.rand(n_param)
            self.bestCost = np.inf #Best cost of this particle
            self.bestPosition = np.copy(self.position) #Position of best cost for this particle
            self.currentCost = None
        
        def set_cost(self, cost):
            self.currentCost = cost
            if cost < self.bestCost:
                self.bestCost = cost
                self.bestPosition = np.copy(self.position)
    
    def cost_func(x):
        return bm.cost(transform(x))
    
    def transform(x):
        if bm._bounded:
            xTrans = bm.lb + x*(bm.ub-bm.lb) #Map x to [lb,ub]
        else:
            xTrans = x*2 #Map x from [0,1] to [0,2]
            if not bm._useScaleFactors:
                xTrans = xTrans*bm.defaultParams #Map to [0,2*default]
        return xTrans
    
    if len(groups) == 0:
        groups = [[i] for i in range(bm.n_parameters())]
    
    q = 0 #Number of generations without improvement
    #Generate patterns
    patterns = [] #All combinations of groups
    for i in range(len(groups)):
        for j in itertools.combinations(range(len(groups)),i+1):
            newPattern = []
            for k in j:
                newPattern += groups[k]
            patterns += [newPattern]
    N = len(patterns) #Number of patterns
    
    particleList = []
    for i in range(n):
        particleList.append(particle(bm.n_parameters()))
    
    Gcost = [np.inf]*lmax #Best cost ever
    Gpos = [None]*lmax #Position of best cost ever
    alpha = [None]*lmax
    for L in range(lmax):
        if L > 0:
            Gcost[L] = Gcost[L-1]
            Gpos[L] = Gpos[L-1]
        
        if debug:
            print('-------------')
            print("Begginning population: "+str(L))
            print("Best cost so far: "+str(Gcost[L]))
            print("Found at position: "+str(Gpos[L]))
        
        foundImprovement = False
        for p in particleList:
            cost = cost_func(p.position)
            p.set_cost(cost)
            if cost < Gcost[L]:
                Gcost[L] = cost
                Gpos[L] = np.copy(p.position)
                foundImprovement = True
        
        if foundImprovement:
            q = 0
            if debug:
                print("Found improvement")
                print("Best cost is now: "+str(Gcost[L]))
                print("Found at position: "+str(Gpos[L]))
        else:
            q += 1
            if debug:
                print("Didn't find an improvement")
                print("Current value of q: "+str(q))
        
        if Gcost[L] < gmin:
            print("Cost successfully minimised")
            print("Final cost of:")
            print(Gcost[L])
            print("found at:")
            print(Gpos[L])
            break
        
        if q>5*qmax:
            #Abort
            print("Too many iterations without improvement")
            print("Final cost of:")
            print(Gcost[L])
            print("found at:")
            print(Gpos[L])
            break
        
        if q>=qmax:
            if debug:
                print("q exceeds qmax so perturbing")
                print("q: "+str(q))
                print("qmax: "+str(qmax))
            #steps 5.1 and 5.2
            newParticleList = []
            bestNewCost = np.inf
            bestNewPosition = None
            for i in range(N):
                newParticle = particle(bm.n_parameters())
                newParticle.position = np.copy(Gpos[L])
                for j in patterns[i]:
                    newParticle.position[j] *= 1+(np.random.rand()-0.5)/40
                    if newParticle.position[j] > 1:
                        newParticle.position[j] = 1
                cost = cost_func(newParticle.position)
                newParticle.set_cost(cost)
                if cost<bestNewCost:
                    bestNewCost = cost
                    bestNewPosition = np.copy(newParticle.position)
                newParticleList.append(newParticle)
            if debug:
                print("Perturbed particles")
                print("Best new cost is: "+str(bestNewCost))
                print("Found at: "+str(bestNewPosition))
            if bestNewCost <= Gcost[L]:
                if debug:
                    print("Cost improved by perturbing")
                    print("Original best cost: "+str(Gcost[L]))
                    print("New best cost: "+str(bestNewCost))
                q = 0
                Gcost[L] = bestNewCost
                Gpos[L] = np.copy(bestNewPosition)
                worstCost = -np.inf
                worstPosition = None
                for p in particleList:
                    if p.currentCost > worstCost:
                        worstCost = p.currentCost
                        worstPosition = p.position
                #Found worst cost and position, now to find that particle
                if debug:
                    print("Adjusting worst particle")
                for p in particleList:
                    if p.currentCost == worstCost and all(p.position == worstPosition):
                        if debug:
                            print("Worst particle found")
                            print("Worst particle cost: "+str(worstCost))
                            print("Worst particle position: "+str(worstPosition))
                        p.bestCost = bestNewCost
                        p.bestPosition = np.copy(bestNewPosition)
                        if debug:
                            print("New best cost for worst particle: "+str(p.bestCost))
                            print("New best position for worst particle: "+str(p.bestPosition))
            else:
                if debug:
                    print("Cost wasn't improved by perturbing")
        
        #Step 6
        alpha[L] = 1/n*np.sum(np.abs([p.currentCost-Gcost[L] for p in particleList]))
        if L>=2:
            #Adapt inertia weight w from (17)
            w = np.exp(-alpha[L-1]/alpha[L-2])
        if debug:
            print("Updating inertia")
            print("w: "+str(w))
            print("alpha: "+str(alpha[L]))
        
        #Renew velocities according to (16)
        for p in particleList:
            localAcc = c1*np.random.rand()*(p.bestPosition-p.position)
            globalAcc = c2*np.random.rand()*(Gpos[L]-p.position)
            p.velocity = w*p.velocity + localAcc + globalAcc
        if debug:
            print("Velocities renewed")
        #Move positions according to (15) while maintaining [0,1] bounds
        for p in particleList:
            p.position += p.velocity
            for i in range(bm.n_parameters()):
                if p.position[i] < 0:
                    p.position[i] = 0
                elif p.position[i] > 1:
                    p.position[i] = 1
        
        if debug:
            print("Positions renewed")
            print("Finished population: "+str(L))
            print("Best cost so far: "+str(Gcost[L]))
            print("Found at position: "+str(Gpos[L]))

    bm.evaluate(transform(Gpos[L]))
    return transform(Gpos[L])

if __name__ == '__main__':
    groups = [[0,2,4,6],[1,3,5,7],[8]]
    bm = ionbench.problems.staircase.HH_Benchmarker()
    run(bm,groups,debug=True)

def get_approach():
    """
    The approach in Chen et al 2012 uses bounds. The method also uses scale factors in such a way that the range of parameters is bounded between 0 and 1, however, this is done automatically in the optimisers for bounded benchmarkers.

    Returns
    -------
    app : approach
        The approach used in Chen et al 2012.

    """
    app = ionbench.approach.Chen2012()
    return app

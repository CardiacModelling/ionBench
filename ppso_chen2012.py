import numpy as np
import benchmarker
#Notes: The algorithm defines parameters between 0 and 1, this is mapped to 0 to 2 when the cost function is called

class particle:
    def __init__(self, n_param):
        self.velocity = 0.1*np.random.rand(n_param)
        self.position = np.random.rand(n_param)
        self.bestCost = np.inf #Best cost of this particle
        self.bestPosition = self.position #Position of best cost for this particle
        self.currentCost = None
    
    def setCost(self, cost):
        self.currentCost = cost
        if cost < self.bestCost:
            self.bestCost = cost
            self.bestPosition = self.position

debug = True
n = 20 #Number of particles
c1 = 1.4 #Local acceleration constant
c2 = 1.4 #Global acceleration constant
qmax = 5 #Maximum number of generations of nonimprovement
lmax = 200 #Maximum number of generations
gmin = 0.05 #Minimum fitness
w = 0.6 #Inertia weight

q = 0 #Number of generations without improvement

bm = benchmarker.HH_Benchmarker()

#Generate patterns
groups = [[0,2,4,6],[1,3,5,7],[8]]
patterns = [groups[0],groups[1],groups[2], groups[0]+groups[1], groups[0]+groups[2], groups[1]+groups[2], groups[0]+groups[1]+groups[2]]
N = len(patterns) #Number of patterns

particleList = []
for i in range(n):
    particleList.append(particle(bm.n_parameters()))

Gcost = [np.inf]*lmax #Best cost ever
Gpos = [None]*lmax #Position of best cost ever
alpha = [None]*lmax
for l in range(lmax):
    if l > 0:
        Gcost[l] = Gcost[l-1]
        Gpos[l] = Gpos[l-1]
    
    if debug:
        print('-------------')
        print("Begginning population: "+str(l))
        print("Best cost so far: "+str(Gcost[l]))
        print("Found at position: "+str(Gpos[l]))
    
    foundImprovement = False
    for p in particleList:
        cost = bm.cost(p.position*2)
        p.setCost(cost)
        if cost < Gcost[l]:
            Gcost[l] = cost
            Gpos[l] = p.position
            foundImprovement = True
    
    if foundImprovement:
        q = 0
        if debug:
            print("Found improvement")
            print("Best cost is now: "+str(Gcost[l]))
            print("Found at position: "+str(Gpos[l]))
    else:
        q += 1
        if debug:
            print("Didn't find an improvement")
            print("Current value of q: "+str(q))
    
    if Gcost[l] < gmin:
        print("Cost successfully minimised")
        print("Final cost of:")
        print(Gcost[l])
        print("found at:")
        print(Gpos[l])
        break
    
    if q>5*qmax:
        #Abort
        print("Too many iterations without improvement")
        print("Final cost of:")
        print(Gcost[l])
        print("found at:")
        print(Gpos[l])
        break
    
    if q>=qmax:
        if debug:
            print("q exceeds qmax so perturbing")
            print("q: "+str(q))
            print("qmax: "+str(qmax))
        #steps 5.1 and 5.2
        newParticleList = []
        bestNewCost = np.inf
        for i in range(N):
            newParticle = particle(bm.n_parameters())
            newParticle.position = Gpos[l]
            for j in patterns[i]:
                newParticle.position[j] *= 1+(np.random.rand()-0.5)/40
                if newParticle.position[j] > 1:
                    newParticle.position[j] = 1
            cost = bm.cost(newParticle.position*2)
            newParticle.setCost(cost)
            if cost<bestNewCost:
                bestNewCost = cost
                bestNewPosition = newParticle.position
            newParticleList.append(newParticle)
        if debug:
            print("Perturbed particles")
            print("Best new cost is: "+str(bestNewCost))
            print("Found at: "+str(bestNewPosition))
        if bestNewCost <= Gcost[l]:
            if debug:
                print("Cost improved by perturbing")
                print("+++++++++++++++++++++++++++++++++++++++OMG it worked------------------------------------------")
                print("Original best cost: "+str(Gcost[l]))
                print("New best cost: "+str(bestNewCost))
            q = 0
            Gcost[l] = bestNewCost
            Gpos[l] = bestNewPosition
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
                    p.bestPosition = bestNewPosition
                    if debug:
                        print("New best cost for worst particle: "+str(p.bestCost))
                        print("New best position for worst particle: "+str(p.bestPosition))
        else:
            if debug:
                print("Cost wasn't improved by perturbing")
    
    #Step 6
    alpha[l] = 1/n*np.sum(np.abs([p.currentCost-Gcost[l] for p in particleList]))
    if l>=2:
        #Adapt inertia weight w from (17)
        w = np.exp(-alpha[l-1]/alpha[l-2])
    if debug:
        print("Updating inertia")
        print("w: "+str(w))
        print("alpha: "+str(alpha[l]))
    
    #Renew velocities according to (16)
    for p in particleList:
        localAcc = c1*np.random.rand()*(p.bestPosition-p.position)
        globalAcc = c2*np.random.rand()*(Gpos[l]-p.position)
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
        print("Finished population: "+str(l))
        print("Best cost so far: "+str(Gcost[l]))
        print("Found at position: "+str(Gpos[l]))

bm.evaluate(Gpos[l]*2)
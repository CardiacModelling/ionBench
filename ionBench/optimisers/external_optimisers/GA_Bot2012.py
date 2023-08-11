import numpy as np
from ionBench.problems import staircase
from functools import cache

from pymoo.core.individual import Individual
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.core.population import Population

def run(bm, nGens = 50, eta_cross = 10, eta_mut = 20, popSize = 50, debug = False):
    """
    Runs the genetic algorithm from Bot et al 2012.

    Parameters
    ----------
    bm : Benchmarker
        A benchmarker to evaluate the performance of the optimisation algorithm.
    nGens : int, optional
        The number of generations to run the optimisation algorithm for. The default is 50.
    eta_cross : float, optional
        Crossover parameter. The default is 10.
    eta_mut : float, optional
        Mutation parameter. The default is 20.
    popSize : int, optional
        The size of the population in each generation. The default is 50.
    debug : bool, optional
        If True, debug information will be printed, reporting that status of the optimisation each generation. The default is False.

    Returns
    -------
    xbest : list
        The best parameters identified.

    """
    class individual():
        def __init__(self):
            self.x = bm.sample()
            self.cost = None
        def findCost(self):
            self.cost = costFunc(tuple(self.x))
    
    @cache
    def costFunc(x):
        return bm.cost(x)
    
    pop = [None]*popSize
    for i in range(popSize):
        pop[i] = individual()
        pop[i].findCost()
    
    for gen in range(nGens):
        minCost = np.inf
        for i in range(popSize):
            if pop[i].cost < minCost:
                minCost = pop[i].cost
                elite = pop[i]
        if debug:
            print("------------")
            print("Gen "+str(gen))
            print("Best cost: "+str(minCost))
        #Tournement selection
        newPop = []
        for j in range(2):
            perm = np.random.permutation(popSize)
            for i in range(popSize//2):
                if pop[perm[2*i]].cost > pop[perm[2*i+1]].cost:
                    newPop.append(pop[perm[2*i]])
                else:
                    newPop.append(pop[perm[2*i+1]])
        pop = newPop #Population of parents
        #Crossover SBX
        newPop = []
        problem = Problem(n_var=bm.n_parameters(), xl=0.0, xu=2.0)
        for i in range(popSize//2):
            a, b = Individual(X=np.array(pop[2*i].x)), Individual(X=np.array(pop[2*i+1].x))
    
            parents = [[a, b]]
            off = SBX(prob=0.9, eta=eta_cross).do(problem, parents) #What is prob vs prob_var
            Xp = off.get("X")
            newPop.append(individual())
            newPop[-1].x = Xp[0]
            newPop.append(individual())
            newPop[-1].x = Xp[1] #Can this be done in one line
        pop = newPop
        #Mutation
        mutation = PolynomialMutation(prob=0.1*bm.n_parameters(), eta=eta_mut)
        for i in range(popSize):
            ind = Population.new(X=[pop[i].x])
            off = mutation(problem, ind)
            pop[i].x = off.get("X")[0]
        if debug:
            print("Finishing gen "+str(gen))
        #Find costs
        for i in range(popSize):
            pop[i].findCost()
        #Elitism
        maxCost = -np.inf
        for i in range(popSize):
            if pop[i].cost > maxCost:
                maxCost = pop[i].cost
                maxIndex = i
        pop[maxIndex] = elite
    
    minCost = np.inf
    for i in range(popSize):
        if pop[i].cost < minCost:
            minCost = pop[i].cost
            elite = pop[i]
    bm.evaluate(elite.x)
    return elite.x

if __name__ == '__main__':
    bm = staircase.HH_Benchmarker()
    run(bm, debug=True)
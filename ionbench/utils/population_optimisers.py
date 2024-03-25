import numpy as np
import copy

from pymoo.core.individual import Individual as pymooInd
from pymoo.core.population import Population
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.crossover.pntx import SinglePointCrossover
from pymoo.operators.mutation.pm import PolynomialMutation


class Individual:
    """
    Individual class for genetic algorithms.
    """

    def __init__(self, bm, x0, cost_func):
        """
        Initialise an individual.
        Parameters
        ----------
        bm : ionbench.Benchmarker
            Benchmarker object.
        x0 : np.ndarray
            Initial guess for the population. Individual will be sampled around x0.
        cost_func : function
            Cost function.
        """
        if x0 is None:
            self.x = bm.sample()
        else:
            self.x = bm.input_parameter_space(
                bm.original_parameter_space(x0) * np.random.uniform(low=0.5, high=1.5, size=bm.n_parameters()))
        self.x = bm.clamp_parameters(self.x)
        self.cost = None
        self.cost_func = cost_func

    def find_cost(self):
        """
        Find the cost of the individual for its current self.x. Updates self.cost.
        """
        self.cost = self.cost_func(tuple(self.x))


def get_pop(bm, x0, n, cost_func):
    """
    Generate a population of n individuals.

    Parameters
    ----------
    bm : ionbench.Benchmarker
        Benchmarker object.
    x0 : np.ndarray
        Initial guess for the population to generate around.
    n : int
        Number of individuals to generate.
    cost_func : function
        Cost function.

    Returns
    -------
    pop : list
        List of individuals.
    """
    pop = [None] * n
    for i in range(n):
        pop[i] = Individual(bm, x0, cost_func)
    return find_pop_costs(pop)


def find_pop_costs(pop):
    """
    Find the costs of the population by applying pop[i].find_cost() to all individuals.

    Parameters
    ----------
    pop : list
        List of individuals.

    Returns
    -------
    pop : list
        List of individuals with costs updated.
    """
    for i in range(len(pop)):
        pop[i].find_cost()
    return pop


def tournament_selection(pop):
    """
    Perform tournament selection on the population.

    Parameters
    ----------
    pop : list
        List of individuals.

    Returns
    -------
    newPop : list
        New population after tournament selection.
    """
    newPop = []
    for j in range(2):
        perm = np.random.permutation(len(pop))
        for i in range(len(pop) // 2):
            if pop[perm[2 * i]].cost < pop[perm[2 * i + 1]].cost:
                newPop.append(copy.deepcopy(pop[perm[2 * i]]))
            else:
                newPop.append(copy.deepcopy(pop[perm[2 * i + 1]]))
    return newPop  # Population of parents


def one_point_crossover(pop, bm, cost_func, crossoverProb=0.5):
    """
    Perform one point crossover on the population.

    Parameters
    ----------
    pop : list
        List of individuals
    bm : ionbench.Benchmarker
        Benchmarker object
    cost_func : function
        Cost function
    crossoverProb : float, optional
        Probability of crossover. The default is 0.5.

    Returns
    -------
    newPop : list
        New population after crossover.
    """
    newPop = []
    problem = Problem(n_var=bm.n_parameters(), xl=bm.input_parameter_space(bm.lb), xu=bm.input_parameter_space(bm.ub))
    for i in range(len(pop) // 2):
        a, b = pymooInd(X=np.array(pop[2 * i].x)), pymooInd(X=np.array(pop[2 * i + 1].x))
        parents = [[a, b]]
        off = SinglePointCrossover(prob=crossoverProb).do(problem, parents)
        newPop = add_pymoo(bm, newPop, off, cost_func)
    return newPop


def add_pymoo(bm, pop, off, cost_func):
    """
    Add the pymoo population to the ionbench population list.
    Parameters
    ----------
    bm : ionbench.Benchmarker
        Benchmarker object
    pop : list
        List of individuals (ionbench population)
    off : pymoo.core.population.Population
        Population of individuals (pymoo population)
    cost_func : function
        A cost function to use for the ionbench individuals.

    Returns
    -------
    pop : list
        List of individuals (ionbench population) with the new individuals added.
    """
    Xp = off.get("X")
    for i in range(len(Xp)):
        pop.append(Individual(bm, bm.sample(), cost_func))
        pop[-1].x = Xp[i]
    return pop


def sbx_crossover(pop, bm, cost_func, eta_cross):
    """
    Perform one point crossover on the population.

    Parameters
    ----------
    pop : list
        List of individuals
    bm : ionbench.Benchmarker
        Benchmarker object
    cost_func : function
        Cost function
    eta_cross : float
        Parameter for the crossover.

    Returns
    -------
    newPop : list
        New population after crossover.
    """
    newPop = []
    problem = Problem(n_var=bm.n_parameters(), xl=bm.input_parameter_space(bm.lb), xu=bm.input_parameter_space(bm.ub))
    for i in range(len(pop) // 2):
        a, b = pymooInd(X=np.array(pop[2 * i].x)), pymooInd(X=np.array(pop[2 * i + 1].x))
        parents = [[a, b]]
        off = SBX(prob=0.9, prob_var=0.5, eta=eta_cross).do(problem, parents)
        newPop = add_pymoo(bm, newPop, off, cost_func)
    return newPop


def polynomial_mutation(pop, bm, cost_func, eta_mut):
    """
    Perform polynomial mutation from Bot et al. 2012 on the population.

    Parameters
    ----------
    pop : list
        List of individuals
    bm : ionbench.Benchmarker
        Benchmarker object
    cost_func : function
        Cost function
    eta_mut : float
        Parameter for the mutation.

    Returns
    -------
    newPop : list
        New population after crossover.
    """
    problem = Problem(n_var=bm.n_parameters(), xl=bm.input_parameter_space(bm.lb), xu=bm.input_parameter_space(bm.ub))
    mutation = PolynomialMutation(prob=1, prob_var=0.1, eta=eta_mut)
    for i in range(len(pop)):
        ind = Population.new(X=[pop[i].x])
        off = mutation(problem, ind)
        pop[i] = Individual(bm, bm.sample(), cost_func)
        pop[i].x = off.get("X")[0]
    return pop


def get_elites(pop, n):
    """
    Get the n elite individuals from the population.

    Parameters
    ----------
    pop : list
        List of individuals
    n : int
        Number of elite individuals to return

    Returns
    -------
    list
        List of elite individuals
    """
    costVec = [pop[i].cost for i in range(len(pop))]
    eliteIndices = np.argsort(costVec)[:n]
    elites = [copy.deepcopy(pop[eliteIndices[i]]) for i in range(n)]
    return elites


def set_elites(pop, elites):
    """
    Set the elite individuals in the population.

    Parameters
    ----------
    pop : list
        List of individuals.
    elites : list
        List of elite individuals.

    Returns
    -------
    pop : list
        Copy of the list of individuals (pop) with the elite individuals replacing the worst individuals.
    """
    costVec = [pop[i].cost for i in range(len(pop))]
    eliteIndices = np.argsort(costVec)[-len(elites):]
    for i in range(len(elites)):
        pop[eliteIndices[i]] = copy.deepcopy(elites[i])
    return copy.deepcopy(pop)

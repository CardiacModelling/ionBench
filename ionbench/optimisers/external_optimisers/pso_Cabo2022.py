import numpy as np
import ionbench


def run(bm, x0=[], maxIter=1000, gmin=0.05, debug=False):
    """
    Runs the PSO algorithm defined by Cabo 2022.

    Parameters
    ----------
    bm : Benchmarker
        A benchmarker to evaluate the performance of the optimisation algorithm.
    x0 : list, optional
        Initial parameter guess. Population is generated by randomly perturbing this initial guess +-50%, then applying the appropriate bounds. If x0=[] (the default), then the population will be sampled uniformly between the bounds.
    maxIter : int, optional
        Maximum number of iterations. The default is 1000.
    debug : bool, optional
        If True, debug information will be printed, reporting that status of the optimisation each generation. The default is False.

    Returns
    -------
    xbest : list
        The best parameters identified.

    """
    if len(x0) == 0:
        x0 = bm.sample()

    class Particle:
        def __init__(self):
            self.position = bm.input_parameter_space(bm.original_parameter_space(x0) * np.random.uniform(low=0.5, high=1.5, size=bm.n_parameters()))
            self.position = bm.clamp(self.position)
            self.velocity = (bm.input_parameter_space(bm.original_parameter_space(x0) * np.random.uniform(low=0.5, high=1.5, size=bm.n_parameters())) - self.position) / 2
            self.bestCost = np.inf  # Best cost of this particle
            self.bestPosition = np.copy(self.position)  # Position of best cost for this particle
            self.currentCost = None

        def set_cost(self, cost):
            self.currentCost = cost
            if cost < self.bestCost:
                self.bestCost = cost
                self.bestPosition = np.copy(self.position)

    def cost_func(x):
        return bm.cost(x)

    lb = bm.input_parameter_space(bm.lb)
    ub = bm.input_parameter_space(bm.ub)

    # Set population size n
    n = int(10 + 2 * np.sqrt(bm.n_parameters()))
    # Initial population
    particleList = []
    for i in range(n):
        particleList.append(Particle())

    Gcost = [np.inf] * maxIter  # Best cost ever
    Gpos = [None] * maxIter  # Position of best cost ever
    for L in range(maxIter):
        if L > 0:
            Gcost[L] = Gcost[L - 1]
            Gpos[L] = Gpos[L - 1]

        if debug:
            print('-------------')
            print(f'Begginning population: {L}')
            print(f'Best cost so far: {Gcost[L]}')
            print(f'Found at position: {Gpos[L]}')

        for p in particleList:
            cost = cost_func(p.position)
            p.set_cost(cost)
            if cost < Gcost[L]:
                Gcost[L] = cost
                Gpos[L] = np.copy(p.position)

        if Gcost[L] < gmin:
            print("Cost successfully minimised")
            print(f'Final cost of {Gcost[L]} found at:')
            print(Gpos[L])
            break

        # Renew velocities
        c1 = 1.496  # Assume they used the fixed value
        c2 = 1.496  # Assume they used the fixed value
        w = 0.7298  # Assume they used the fixed value
        for p in particleList:
            localAcc = c1 * np.random.rand() * (p.bestPosition - p.position)
            globalAcc = c2 * np.random.rand() * (Gpos[L] - p.position)
            p.velocity = w * p.velocity + localAcc + globalAcc
        if debug:
            print("Velocities renewed")
        # Move positions
        for p in particleList:
            p.position += p.velocity
            # Enfore bounds by clamping
            if not bm.in_bounds(bm.original_parameter_space(p.position)):
                for i in range(bm.n_parameters()):
                    if p.position[i] > ub[i]:
                        p.position[i] = ub[i]
                        p.velocity[i] = 0
                    elif p.position[i] < lb[i]:
                        p.position[i] = lb[i]
                        p.velocity[i] = 0

        if debug:
            print("Positions renewed")
            print(f'Finished population {L}')
            print(f'Best cost so far: {Gcost[L]}')
            print(f'Found at position: {Gpos[L]}')

    bm.evaluate(Gpos[L])
    return Gpos[L]


def get_modification(modNum=1):
    """
    modNum = 1 -> Cabo2022

    Returns
    -------
    mod : modification
        Modification corresponding to inputted modNum. Default is modNum = 1, so Cabo2022.

    """
    mod = ionbench.modification.Cabo2022()
    return mod


if __name__ == '__main__':
    bm = ionbench.problems.staircase.HH_Benchmarker()
    mod = get_modification()
    mod.apply(bm)
    run(bm, maxIter=200, gmin=0.02, debug=True, **mod.kwargs)

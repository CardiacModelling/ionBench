"""
This module contains the PSO algorithm defined by Seemann et al. 2009.
The initial sampling procedure for particle positions and velocities is not described. The ionBench default are used here.
For the acceleration constants, c1 and c2. A linearly decreasing c1 and increasing c2 was recommended so that is used here.
For the inertia weight, w, either exponentially decreasing or randomly sampled in [0.5,1] was recommended. Since the rate of the exponential decrease was not described, random sampling is used here.
There are no bounds on parameters for Seemann et al. 2009 PSO, so the parameters are not transformed to [0,1] and the particle positions are not clamped.
"""
import numpy as np
import ionbench


# noinspection PyShadowingNames
def run(bm, x0=None, n=20, maxIter=1000, debug=False):
    """
    Runs the PSO algorithm defined by Seemann et al. 2009.

    Parameters
    ----------
    bm : Benchmarker
        A benchmarker to evaluate the performance of the optimisation algorithm.
    x0 : list, optional
        Initial parameter guess around which the population is sampled. If x0=None (the default), then x0 will be sampled with bm.sample().
    n : int, optional
        Number of particles. The default is 20.
    maxIter : int, optional
        Maximum number of iterations. The default is 1000.
    debug : bool, optional
        If True, debug information will be printed, reporting that status of the optimisation each generation. The default is False.

    Returns
    -------
    xbest : list
        The best parameters identified.

    """
    if x0 is None:
        x0 = bm.sample()

    cost_func = ionbench.utils.cache.get_cached_cost(bm)

    # noinspection PyShadowingNames
    class Particle(ionbench.utils.particle_optimisers.Particle):
        """
        Seemann et al. 2009 does not use parameter bounds so cannot be mapped to [0,1]. We implement this by overriding the transform and clamp methods.
        """
        def __init__(self):
            super().__init__(bm, cost_func, x0)

        def clamp(self):
            pass

        def transform(self, parameters):
            return parameters

        def untransform(self, parameters):
            return parameters

    particleList = []
    for i in range(n):
        particleList.append(Particle())

    Gcost = [np.inf] * maxIter  # Best cost ever
    Gpos = [None] * maxIter  # Position of best cost ever
    L = 0
    for L in range(maxIter):
        if L > 0:
            Gcost[L] = Gcost[L - 1]
            Gpos[L] = Gpos[L - 1]

        if debug:
            print('-------------')
            print(f'Beginning population: {L}')
            print(f'Best cost so far: {Gcost[L]}')
            print(f'Found at position: {Gpos[L]}')

        for p in particleList:
            p.set_cost()
            if p.currentCost < Gcost[L]:
                Gcost[L] = p.currentCost
                Gpos[L] = np.copy(p.position)

        # Renew velocities
        if L < 1000:  # Dependence of c1 and c2 on maxIter has been removed
            c1 = 2.5 - L / 1000 * 2
        else:  # pragma: no cover
            c1 = 0.5
        c2 = 3 - c1
        w = np.random.uniform(0.5, 1)
        for p in particleList:
            localAcc = c1 * np.random.rand() * (p.bestPosition - p.position)
            globalAcc = c2 * np.random.rand() * (Gpos[L] - p.position)
            p.velocity = w * p.velocity + localAcc + globalAcc
        if debug:
            print("Velocities renewed")
        # Move positions
        for p in particleList:
            p.position += p.velocity

        if debug:
            print("Positions renewed")
            print(f'Finished population {L}')
            print(f'Best cost so far: {Gcost[L]}')
            print(f'Found at position: {Gpos[L]}')

        if bm.is_converged():
            break

    if L >= maxIter-1:
        bm.set_max_iter_flag()

    bm.evaluate()
    return Particle().untransform(Gpos[L])


# noinspection PyUnusedLocal,PyShadowingNames
def get_modification(modNum=1):
    """
    modNum = 1 -> Seemann2009

    Returns
    -------
    mod : modification
        Modification corresponding to inputted modNum. Default is modNum = 1, so Seemann2009.

    """
    mod = ionbench.modification.Seemann2009()
    return mod

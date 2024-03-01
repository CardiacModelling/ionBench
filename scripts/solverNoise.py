import ionbench
import numpy as np
import matplotlib.pyplot as plt
import time
import myokit

# Does sampling need better tolerances than the default? Yes, by a lot. Some points are off by 1e-4 whereas default are off by 4e-10.
# Does the gradient need to be more accurate?
bm = ionbench.problems.staircase.HH()
if 'loewe' in bm._name or 'moreno' in bm._name:
    bm.sim = myokit.Simulation(bm.model, bm.protocol())
if 'moreno' in bm._name:
    bm._name = ''  # This is a hack to avoid moreno set_parameters doing something that assumes it's an analytical simulation.


def cost_noise(bm, p):
    """
    Calculate the standard deviation of the cost function (ODE solver noise) for a given parameter vector p.
    """
    costs = []
    for x in range(10):
        trialP = np.copy(p)
        trialP[0] *= 1 + (x - 100 / 2) / 100 * 1e-12
        costs.append(bm.cost(trialP))
    return np.std(costs)


def tol_error(bm, points, abstol, reltol):
    """
    Test the tolerances on the ODE solver for a given problem and set of points. Find the average standard deviation of the cost function (ODE solver noise) around the given points.

    Parameters
    ----------
    bm : benchmarker
        Benchmarker problem to test the tolerances on. Should use an ODE solver for the cost function.
    points : list
        List of parameter vectors to test the tolerances on.
    abstol : float
        Absolute tolerance for the ODE solver.
    reltol : float
        Relative tolerance for the ODE solver.

    Returns
    -------
    avgError : float
        Average standard deviation of the cost function (ODE solver noise).
    """
    bm.reset()
    bm.sim.set_tolerance(abstol, reltol)
    errors = []
    for i in range(len(points)):
        errors.append(cost_noise(bm, points[i]))
    avgError = np.mean(errors)
    return avgError


points = bm.sample(25)
bestTime = np.inf
bestTol = (None, None)
for abstol in [10 ** -i for i in range(7, 13)]:
    for reltol in [10 ** -i for i in range(7, 13)]:
        start = time.time()
        error = tol_error(bm, points, abstol, reltol)
        end = time.time()
        print(f'abstol: {abstol}, reltol: {reltol}, time: {end - start}, error: {error}')
        if end - start > bestTime:
            print('Time was worse than the previous best. Skipping the remaining reltols.')
            break
        if error < 1e-9:
            print(f'Done. abstol: {abstol}, reltol: {reltol}, time: {end - start}')
            if end - start < bestTime:
                bestTime = end - start
                bestTol = (abstol, reltol)
            break
print(bestTol)
print(bestTime)

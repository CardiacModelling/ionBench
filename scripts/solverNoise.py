import ionbench
import numpy as np
import matplotlib.pyplot as plt
import time
import myokit
import warnings


def cost_noise(bm, p, nPoints):
    """
    Calculate the standard deviation of the cost function (ODE solver noise) for a given parameter vector p.
    """
    costs = []
    for x in range(nPoints):
        trialP = np.copy(p)
        trialP[0] *= 1 + (x - nPoints / 2) / nPoints * 1e-12
        costs.append(bm.cost(trialP))
    return np.std(costs)


def tol_error(bm, points, abstol, reltol, nPoints):
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
    nPoints : int
        Number of points to test around each parameter vector.

    Returns
    -------
    lowError : float
        25th percentile of the standard deviation of the cost function (ODE solver noise).
    medError : float
        Average (median) standard deviation of the cost function (ODE solver noise).
    highError : float
        75th percentile of the standard deviation of the cost function (ODE solver noise).
    """
    bm.reset()
    bm.sim.set_tolerance(abstol, reltol)
    errors = []
    for i in range(len(points)):
        errors.append(cost_noise(bm, points[i], nPoints))
    lowError, medError, highError = np.nanpercentile(errors, [25, 50, 75])
    return lowError, medError, highError


warnings.simplefilter('ignore')
bm = ionbench.problems.staircase.HH()

abstolBounds = (5, 8)
reltolBounds = (5, 8)
print(
    f'Searching tolerance combinations for {bm.NAME}. Absolute tolerances between 1e-{abstolBounds[0]} and 1e-{abstolBounds[1]}. Relative tolerances between 1e-{reltolBounds[0]} and 1e-{reltolBounds[1]}.')

if 'loewe' in bm.NAME or 'moreno' in bm.NAME:
    bm.sim = myokit.Simulation(bm.model, bm.protocol())
if 'moreno' in bm.NAME:
    bm.NAME = ''  # This is a hack to avoid moreno set_parameters doing something that assumes it's an analytical simulation.

# Loop through lots of tolerance combinations and see what works
points = bm.sample(25)
output = []  # List of tuples (time, abstol, reltol, median error, low error, high error) for successes
for abstol in [10 ** -i for i in range(abstolBounds[0], abstolBounds[1] + 1)]:
    for reltol in [10 ** -i for i in range(reltolBounds[0], reltolBounds[1] + 1)]:
        start = time.time()
        le, me, he = tol_error(bm, points, abstol, reltol, nPoints=10)
        end = time.time()
        if me < 1e-9 and he < 1e-8:
            print(
                f'Success. abstol: {abstol}, reltol: {reltol}, time: {end - start:.2f}, median error: {me:.2e}, error 25th percentile: {le:.2e}, error 75th percentile: {he:.2e}')
            output.append((end - start, abstol, reltol, me, le, he))
        else:
            print(
                f'Fail. abstol: {abstol}, reltol: {reltol}, time: {end - start:.2f}, median error: {me:.2e}, error 25th percentile: {le:.2e}, error 75th percentile: {he:.2e}')

# Print all the valid tolerance combinations with timings and ask the user to choose.
while True:
    print('Tolerance Combinations that work:')
    for t, a, r, me, le, he in output:
        print(
            f'abstol: {a}, reltol: {r}, time: {t:.2f}, median error: {me:.2e}, error 25th percentile: {le:.2e}, error 75th percentile: {he:.2e}')
    a = input('Choose an abstol:\n')
    b = input('Choose a reltol:\n')
    bestTol = (float(a), float(b))

    # How long does the cost and the gradient take to calculate with these tolerances across the whole parameter space?
    print('Checking timing.')
    n = 20
    bm.reset()
    bm.use_sensitivities()
    bm.simSens.set_tolerance(bestTol[0], bestTol[1])
    p = bm.sample(n)
    start = time.time()
    for i in p:
        bm.grad(i)
        print(i)
    end = time.time()
    print(f'Gradient time: {(end - start) / n:.4f}')

    bm.sim.set_tolerance(bestTol[0], bestTol[1])
    start = time.time()
    for i in p:
        bm.cost(i)

    end = time.time()
    print(f'Cost time: {(end - start) / n:.4f}')

    # What is the accuracy of the cost function over the whole parameter space?
    print('Checking solver noise very carefully.')
    le, me, he = tol_error(bm, bm.sample(n=100), bestTol[0], bestTol[1], nPoints=10)
    print(
        f'Median error across parameter space: {me:.2e}, error 25th percentile: {le:.2e}, error 75th percentile: {he:.2e}')
    if me < 1e-9 and he < 1e-8:
        print(f'Error successfully stayed below limits so these tolerances {bestTol} are acceptable.')
        break
    else:
        print(
            'When checking the tolerances very carefully, the error was too high. Please choose another set of tolerances.')

# Solver noise at the default parameters is important as it is influences the likelihood plots
print('Checking solver noise at default parameters.')
error = cost_noise(bm, bm.defaultParams, 100)
print(f'Error at default parameters: {error:.2e}')

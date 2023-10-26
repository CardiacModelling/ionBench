import ionbench
import numpy as np

# %% Profile likelihood


def var(x):
    return np.linspace(1 - x, 1 + x, 2)


variations = [var(0.2)] * 4 + [var(0.4)] * 4 + [var(0.2)]  # aVar = 0.2, bVar = 0.4
bm = ionbench.problems.staircase.HH_Benchmarker()
ionbench.uncertainty.profile_likelihood.run(bm, variations, filename='hh')

variations = [var(0.8), var(0.8), var(0.8), var(0.8), var(0.8), var(0.8), var(0.8), var(0.8), var(0.8), var(0.8), var(0.8), var(0.8), var(0.8), var(0.8)]
variations = [[1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], var(0.8), var(0.8), var(0.8)]
bm = ionbench.problems.staircase.MM_Benchmarker()
ionbench.uncertainty.profile_likelihood.run(bm, variations, filename='mm')

variations = [var(0.99), var(2), var(1.5), var(2), var(1.5), var(1.2), var(0.99), var(0.99), var(0.99), var(0.6), var(0.7), var(0.7)]
bm = ionbench.problems.loewe2016.ikr()
ionbench.uncertainty.profile_likelihood.run(bm, variations, filename='ikr')

variations = [var(0.99), var(2), var(0.99), var(2), var(0.99), var(2), var(2), var(0.99), var(2), var(0.99), var(0.99), var(0.99), var(2), var(2), var(0.99), var(2), var(2), var(2), var(0.99), var(0.99), var(2), var(0.99), var(2), var(0.99), var(0.99)]
bm = ionbench.problems.loewe2016.ikur()
ionbench.uncertainty.profile_likelihood.run(bm, variations, filename='ikur')

# %% FIM
bm = ionbench.problems.staircase.HH_Benchmarker()
# ionbench.uncertainty.fim.explore_solver_noise(bm)
mat = ionbench.uncertainty.fim.run(bm, sigma=1, preoptimise=True, ftol=5e-6, step=1e-4, buffer=1e-4)

bm = ionbench.problems.staircase.MM_Benchmarker()
# ionbench.uncertainty.fim.explore_solver_noise(bm)
mat = ionbench.uncertainty.fim.run(bm, sigma=1, preoptimise=True, ftol=3e-7, step=1e-4, buffer=1e-4)

bm = ionbench.problems.loewe2016.ikr()
# ionbench.uncertainty.fim.explore_solver_noise(bm)
mat = ionbench.uncertainty.fim.run(bm, sigma=1, preoptimise=True, ftol=1e-9, step=1e-6, buffer=1e-8)

bm = ionbench.problems.loewe2016.ikur()
# ionbench.uncertainty.fim.explore_solver_noise(bm)
mat = ionbench.uncertainty.fim.run(bm, sigma=1, preoptimise=True, ftol=1e-9, step=1e-6, buffer=1e-8)

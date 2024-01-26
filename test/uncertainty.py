import ionbench
import numpy as np

# %% Profile likelihood


def var(x):
    return np.linspace(1 - x, 1 + x, 51)

bm = ionbench.problems.staircase.HH_Benchmarker()
variations = [var(0.2)]*bm.n_parameters()
ionbench.uncertainty.profile_likelihood.run(bm, variations, filename='hh')
ionbench.uncertainty.profile_likelihood.run(bm, variations, backwardPass=True, filename='hh')

bm = ionbench.problems.staircase.MM_Benchmarker()
variations = [var(0.2)]*bm.n_parameters()
ionbench.uncertainty.profile_likelihood.run(bm, variations, filename='mm')
ionbench.uncertainty.profile_likelihood.run(bm, variations, backwardPass=True, filename='mm')

bm = ionbench.problems.loewe2016.ikr()
variations = [var(0.2)]*bm.n_parameters()
ionbench.uncertainty.profile_likelihood.run(bm, variations, filename='ikr')
ionbench.uncertainty.profile_likelihood.run(bm, variations, backwardPass=True, filename='ikr')

bm = ionbench.problems.loewe2016.ikur()
variations = [var(0.2)]*bm.n_parameters()
ionbench.uncertainty.profile_likelihood.run(bm, variations, filename='ikur')
ionbench.uncertainty.profile_likelihood.run(bm, variations, backwardPass=True, filename='ikur')

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

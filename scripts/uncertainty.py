import ionbench
import numpy as np


# %% Profile likelihood
def var(x):
    return np.linspace(1 - x, 1 + x, 51)


bm = ionbench.problems.staircase.HH()
variations = [var(0.2)] * bm.n_parameters()
ionbench.uncertainty.profile_likelihood.run(bm, variations, filename='hh')
ionbench.uncertainty.profile_likelihood.run(bm, variations, backwardPass=True, filename='hh')

bm = ionbench.problems.staircase.MM()
variations = [var(0.2)] * bm.n_parameters()
ionbench.uncertainty.profile_likelihood.run(bm, variations, filename='mm')
ionbench.uncertainty.profile_likelihood.run(bm, variations, backwardPass=True, filename='mm')

bm = ionbench.problems.loewe2016.IKr()
variations = [var(0.2)] * bm.n_parameters()
ionbench.uncertainty.profile_likelihood.run(bm, variations, filename='ikr')
ionbench.uncertainty.profile_likelihood.run(bm, variations, backwardPass=True, filename='ikr')

bm = ionbench.problems.loewe2016.IKur()
variations = [var(0.2)] * bm.n_parameters()
ionbench.uncertainty.profile_likelihood.run(bm, variations, filename='ikur')
ionbench.uncertainty.profile_likelihood.run(bm, variations, backwardPass=True, filename='ikur')



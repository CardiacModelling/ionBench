import ionbench
import numpy as np
import os

def var(n):
    return np.linspace(0.8, 1.2, n)


script_path = os.path.abspath(os.path.dirname(__file__))
extension = os.path.join(script_path, '..', 'results', 'uncertainty')

bm = ionbench.problems.staircase.HH()
variations = [var(51)] * bm.n_parameters()
ionbench.uncertainty.profile_likelihood.run(bm, variations, filename=extension+'/hh')
ionbench.uncertainty.profile_likelihood.run(bm, variations, backwardPass=True, filename=extension+'/hh')

bm = ionbench.problems.staircase.MM()
variations = [var(51)] * bm.n_parameters()
ionbench.uncertainty.profile_likelihood.run(bm, variations, filename=extension+'/mm')
ionbench.uncertainty.profile_likelihood.run(bm, variations, backwardPass=True, filename=extension+'/mm')

bm = ionbench.problems.loewe2016.IKr()
variations = [var(51)] * bm.n_parameters()
ionbench.uncertainty.profile_likelihood.run(bm, variations, filename=extension+'/ikr')
ionbench.uncertainty.profile_likelihood.run(bm, variations, backwardPass=True, filename=extension+'/ikr')

bm = ionbench.problems.loewe2016.IKur()
variations = [var(51)] * bm.n_parameters()
ionbench.uncertainty.profile_likelihood.run(bm, variations, filename=extension+'/ikur')
ionbench.uncertainty.profile_likelihood.run(bm, variations, backwardPass=True, filename=extension+'/ikur')

bm = ionbench.problems.moreno2016.INa()
variations = [var(1005) if i in [0, 1, 5, 8, 9, 10, 11, 12, 13, 14] else var(103) for i in range(bm.n_parameters())]
ionbench.uncertainty.profile_likelihood.run(bm, variations, filename=extension+'/ina')
ionbench.uncertainty.profile_likelihood.run(bm, variations, backwardPass=True, filename=extension+'/ina')

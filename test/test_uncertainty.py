import os

import ionbench
import numpy as np
import matplotlib.pyplot as plt


def test_profile_likelihood(monkeypatch):
    # Generate profile likelihood plots (don't actually plot). Only checks for crashes/error
    def var(x):
        return np.linspace(1 - x, 1 + x, 3)

    def savefig(*args, **kwargs):
        pass

    monkeypatch.setattr(plt, 'show', lambda: None)
    monkeypatch.setattr(plt, 'savefig', savefig)
    variations = [var(0.2)] * 9
    bm = ionbench.problems.staircase.HH()
    ionbench.uncertainty.profile_likelihood.run(bm, variations, filename='hh')
    ionbench.uncertainty.profile_likelihood.run(bm, variations, filename='hh', backwardPass=True)
    ionbench.uncertainty.profile_likelihood.plot_profile_likelihood(modelType='hh', debug=True)
    ionbench.uncertainty.profile_likelihood.plot_profile_likelihood(modelType='hh', sharey=False, debug=True)
    for i in range(9):
        os.remove(f'hh_param{i}.pickle')
        os.remove(f'hhB_param{i}.pickle')

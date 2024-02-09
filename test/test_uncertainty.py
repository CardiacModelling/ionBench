import ionbench
import numpy as np
import matplotlib.pyplot as plt


def test_profile_likelihood(monkeypatch):
    # Generate profile likelihood plots (dont actually plot). Only checks for crashes/error
    def var(x):
        return np.linspace(1 - x, 1 + x, 3)
    monkeypatch.setattr(plt, 'show', lambda: None)
    variations = [var(0.2)] * 9  # aVar = 0.2, bVar = 0.4
    bm = ionbench.problems.staircase.HH()
    ionbench.uncertainty.profile_likelihood.run(bm, variations)


def test_fim(monkeypatch):
    # Calculate fim. Only checks for crashes/errors
    monkeypatch.setattr(plt, 'show', lambda: None)
    bm = ionbench.problems.loewe2016.IKr()
    ionbench.uncertainty.fim.run(bm, sigma=1, preoptimise=True, ftol=1e-9, step=1e-6, buffer=1e-8)

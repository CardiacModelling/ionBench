import ionbench
import numpy as np
import matplotlib.pyplot as plt


def test_profile_likelihood(monkeypatch):
    # Generate profile likelihood plots (don't actually plot). Only checks for crashes/error
    def var(x):
        return np.linspace(1 - x, 1 + x, 3)
    monkeypatch.setattr(plt, 'show', lambda: None)
    variations = [var(0.2)] * 9  # aVar = 0.2, bVar = 0.4
    bm = ionbench.problems.staircase.HH()
    ionbench.uncertainty.profile_likelihood.run(bm, variations)

import pytest
import ionbench
from ionbench import approach
import numpy as np
import matplotlib.pyplot as plt

def test_profile_likelihood(monkeypatch):
    monkeypatch.setattr(plt, 'show', lambda: None)
    var = lambda x:np.linspace(1-x,1+x,3)
    variations = [var(0.2)]*9 #aVar = 0.2, bVar = 0.4
    bm = ionbench.problems.staircase.HH_Benchmarker()
    ionbench.uncertainty.profile_likelihood.run(bm, variations)

def test_fim(monkeypatch):
    monkeypatch.setattr(plt, 'show', lambda: None)
    bm = ionbench.problems.loewe2016.ikr()
    mat = ionbench.uncertainty.fim.run(bm, sigma = 1, preoptimise = True, ftol = 1e-9, step = 1e-6, buffer = 1e-8)

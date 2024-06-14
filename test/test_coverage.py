# Extra tests to provide full coverage over ionBench where other tests do not reach
from importlib import import_module
import numpy as np
import ionbench
import pytest


# Problems
def test_problems_coverage():
    bm = ionbench.problems.loewe2016.IKr()
    bm.solve_model(np.arange(0, bm.T_MAX, bm.TIMESTEP), continueOnError=False)
    bm.grad(bm.sample())  # Triggers use_sensitivities() through grad
    bm = ionbench.problems.moreno2016.INa()
    bm.solve_model(np.arange(0, bm.T_MAX, bm.TIMESTEP), continueOnError=False)
    bm = ionbench.problems.test.Test()
    assert len(bm.sample(5)) == 5


# Optimisers
def test_scipy_optimisers():
    """
    Two scipy optimisers aren't fully covered by their modifications, so we add tests here.
    """
    bm = ionbench.problems.test.Test()
    mod = ionbench.modification.Modification(parameterBounds='off', rateBounds='on')
    mod.apply(bm)
    ionbench.optimisers.scipy_optimisers.slsqp_scipy.run(bm, maxIter=10, debug=False)
    bm.reset(False)
    mod = ionbench.modification.Modification(parameterBounds='on')
    mod.apply(bm)
    ionbench.optimisers.scipy_optimisers.powell_scipy.run(bm, maxIter=10, debug=False)


def test_convergence_terminates():
    """
    Test that, in all external optimisers, if the cost threshold is reached, the optimisation terminates.
    """
    bm = ionbench.problems.test.Test()
    for opt in ionbench.OPT_EXT:
        module = import_module(opt)
        module.run(bm)
        assert bm.is_converged()


def test_abort_incompatible_modifications():
    """
    If the modification is incompatible with the optimiser, the optimiser should abort.
    """
    bm = ionbench.problems.test.Test()
    with pytest.raises(RuntimeError):
        ionbench.optimisers.external_optimisers.PSO_Loewe2016.run(bm)
    with pytest.raises(RuntimeError):
        ionbench.optimisers.external_optimisers.PSOTRR_Loewe2016.run(bm)
    with pytest.raises(RuntimeError):
        ionbench.optimisers.external_optimisers.hybridPSOTRR_Loewe2016.run(bm)
    with pytest.raises(RuntimeError):
        ionbench.optimisers.external_optimisers.hybridPSOTRRTRR_Loewe2016.run(bm)


def test_loewe2016():
    """
    Checks Loewe PSO warns about poor choice of phi1 and phi2.
    """
    bm = ionbench.problems.test.Test()
    bm.add_parameter_bounds()
    with pytest.warns(UserWarning):
        ionbench.optimisers.external_optimisers.PSO_Loewe2016.run(bm, phi1=1, phi2=1, maxIter=5)
    with pytest.warns(UserWarning):
        ionbench.optimisers.external_optimisers.PSOTRR_Loewe2016.run(bm, phi1=1, phi2=1, maxIter=5)
    with pytest.warns(UserWarning):
        ionbench.optimisers.external_optimisers.hybridPSOTRR_Loewe2016.run(bm, phi1=1, phi2=1, maxIter=5)
    with pytest.warns(UserWarning):
        ionbench.optimisers.external_optimisers.hybridPSOTRRTRR_Loewe2016.run(bm, phi1=1, phi2=1, maxIter=5)


def test_gurkiewicz2007():
    """
    Checks GA Gurkiewicz warns about using a population size that is too small.
    """
    bm = ionbench.problems.test.Test()
    with pytest.warns(UserWarning):
        ionbench.optimisers.external_optimisers.GA_Gurkiewicz2007a.run(bm, popSize=1, nGens=5)
    with pytest.warns(UserWarning):
        ionbench.optimisers.external_optimisers.GA_Gurkiewicz2007b.run(bm, popSize=1, nGens=5)


def test_cairns2017():
    """
    Checks GA Cairns returns an error when not using bounded parameters.
    """
    bm = ionbench.problems.test.Test()
    with pytest.raises(RuntimeError):
        ionbench.optimisers.external_optimisers.GA_Cairns2017.run(bm)


def test_SA_Vanier_long():
    bm = ionbench.problems.test.Test()
    mod = ionbench.optimisers.external_optimisers.SA_Vanier1999.get_modification()
    mod.apply(bm)
    bm.COST_THRESHOLD = 0
    ionbench.optimisers.external_optimisers.SA_Vanier1999.run(bm, maxIter=100000, debug=True)


def test_PSO_Seemann_long():
    bm = ionbench.problems.test.Test()
    mod = ionbench.optimisers.external_optimisers.pso_Seemann2009.get_modification()
    mod.apply(bm)
    bm.COST_THRESHOLD = 0
    ionbench.optimisers.external_optimisers.pso_Seemann2009.run(bm, maxIter=5000)


def test_PSO_Cabo_long():
    bm = ionbench.problems.test.Test()
    mod = ionbench.optimisers.external_optimisers.pso_Cabo2022.get_modification()
    mod.apply(bm)
    bm.COST_THRESHOLD = 0
    ionbench.optimisers.external_optimisers.pso_Cabo2022.run(bm, maxIter=5000)


def test_Kohjitani_long():
    bm = ionbench.problems.test.Test()
    mod = ionbench.optimisers.external_optimisers.patternSearch_Kohjitani2022.get_modification()
    mod.apply(bm)
    bm.COST_THRESHOLD = 0
    ionbench.optimisers.external_optimisers.patternSearch_Kohjitani2022.run(bm, maxIter=5000)
    bm.reset(False)
    ionbench.optimisers.external_optimisers.patternSearch_Kohjitani2022.run(bm, maxIter=5)
    bm.reset(False)
    ionbench.optimisers.external_optimisers.patternSearch_Kohjitani2022.run(bm, maxfev=5)


def test_NMPSO_Liu_long():
    bm = ionbench.problems.test.Test()
    mod = ionbench.optimisers.external_optimisers.NMPSO_Liu2011.get_modification()
    mod.apply(bm)
    bm.COST_THRESHOLD = 0
    ionbench.optimisers.external_optimisers.NMPSO_Liu2011.run(bm, maxIter=1000, eps=0, debug=True)


def test_Zhou_long():
    bm = ionbench.problems.test.Test()
    mod = ionbench.optimisers.external_optimisers.DE_Zhou2009.get_modification()
    mod.apply(bm)
    bm.COST_THRESHOLD = 0
    ionbench.optimisers.external_optimisers.DE_Zhou2009.run(bm, x0=bm.sample(), debug=True)


def test_hybrid_Loewe_long():
    bm = ionbench.problems.test.Test()
    mod = ionbench.optimisers.external_optimisers.hybridPSOTRR_Loewe2016.get_modification()
    mod.apply(bm)
    bm.COST_THRESHOLD = 0
    ionbench.optimisers.external_optimisers.hybridPSOTRR_Loewe2016.run(bm, x0=bm.sample(), debug=True)
    bm.reset(False)
    ionbench.optimisers.external_optimisers.hybridPSOTRRTRR_Loewe2016.run(bm, x0=bm.sample(), debug=True)


def test_dokos_long():
    bm = ionbench.problems.test.Test()
    # Add noise to get good weight updates
    bm.DATA += np.random.normal(0, 0.003, len(bm.DATA))
    mod = ionbench.optimisers.external_optimisers.curvilinearGD_Dokos2004.get_modification()
    mod.apply(bm)
    bm.COST_THRESHOLD = -1
    ionbench.optimisers.external_optimisers.curvilinearGD_Dokos2004.run(bm, costThreshold=-1, maxIter=250, maxInnerIter=4, debug=True)

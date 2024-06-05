# Extra tests to provide full coverage over ionBench where other tests do not reach
from importlib import import_module
import ionbench
import pytest


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

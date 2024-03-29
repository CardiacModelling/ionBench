import ionbench
from ionbench import modification
import numpy as np
import pytest


class TestModifications:
    bm = ionbench.problems.staircase.HH()

    @pytest.mark.cheap
    def test_bounds(self):
        # Check bounds are correctly applied for all settings
        # No bounds initially
        assert not self.bm.parametersBounded
        # Sampler
        mod = modification.Modification(parameterBounds='on')
        mod.apply(self.bm)
        assert self.bm.parametersBounded
        assert all(np.array(self.bm.lb) > 0)
        assert all(np.array(self.bm.ub) < np.inf)
        # Custom
        lb = np.random.rand(self.bm.n_parameters())
        ub = np.random.rand(self.bm.n_parameters()) + 1
        mod = modification.Modification(parameterBounds='Custom', customBounds=[lb, ub])
        mod.apply(self.bm)
        assert self.bm.parametersBounded
        assert all(self.bm.lb == lb)
        assert all(self.bm.ub == ub)
        # None
        mod = modification.Modification()
        mod.apply(self.bm)
        assert not self.bm.parametersBounded

    @pytest.mark.cheap
    def test_rate_bounds(self):
        # Check rate bounds are correctly applied for all settings
        # No rate bounds initially
        assert not self.bm.ratesBounded
        # On
        mod = modification.Modification(rateBounds='on')
        mod.apply(self.bm)
        assert self.bm.ratesBounded
        # Off
        mod = modification.Modification(rateBounds='off')
        mod.apply(self.bm)
        assert not self.bm.ratesBounded
        # By default, they will be turned off
        mod = modification.Modification()
        mod.apply(self.bm)
        assert not self.bm.ratesBounded

    @pytest.mark.cheap
    def test_log_transform(self):
        # Check log transform is correctly applied for all settings
        # No log transform initially
        assert not any(self.bm.logTransformParams)
        # Standard log transform
        mod = modification.Modification(logTransform='on')
        mod.apply(self.bm)
        assert any(self.bm.logTransformParams)
        assert not all(self.bm.logTransformParams)
        # Custom
        transforms = list(np.random.choice([True, False], self.bm.n_parameters() - 2)) + [True, False]  # Random true and falses in which the last two are True and False
        mod = modification.Modification(logTransform='custom', customLogTransform=transforms)
        mod.apply(self.bm)
        assert any(self.bm.logTransformParams)
        assert all(np.array(self.bm.logTransformParams) == np.array(transforms))
        assert not all(self.bm.logTransformParams)
        # No log transform
        mod = modification.Modification()
        mod.apply(self.bm)
        assert not any(self.bm.logTransformParams)

    @pytest.mark.cheap
    def test_scale_factors(self):
        # Check scale factors are correctly applied
        # No scale factors initially
        assert not self.bm.useScaleFactors
        # Turn on scale factors
        mod = modification.Modification(scaleFactors='on')
        mod.apply(self.bm)
        assert self.bm.useScaleFactors
        # Default is off
        mod = modification.Modification()
        mod.apply(self.bm)
        assert not self.bm.useScaleFactors

    @pytest.mark.cheap
    def test_multiple_settings(self):
        # Check multiple settings applied at once doesn't break anything and order in the .apply() method doesn't matter
        mod = modification.Modification(logTransform='on', parameterBounds='on', scaleFactors='On')
        mod.apply_rate_bounds(mod.dict['rateBounds'], self.bm)
        mod.apply_log_transforms(mod.dict['log transform'], self.bm)
        mod.apply_scale_factors(mod.dict['scale factors'], self.bm)
        mod.apply_parameter_bounds(mod.dict['parameterBounds'], self.bm)
        assert any(self.bm.logTransformParams) and not all(self.bm.logTransformParams)
        assert self.bm.useScaleFactors
        assert self.bm.parametersBounded
        lb = self.bm.lb
        ub = self.bm.ub
        # Check order of applying doesn't change bounds
        emptyMod = modification.Empty()
        emptyMod.apply(self.bm)
        mod = modification.Modification(logTransform='on', parameterBounds='on', scaleFactors='on')
        mod.apply_parameter_bounds(mod.dict['parameterBounds'], self.bm)
        mod.apply_log_transforms(mod.dict['log transform'], self.bm)
        mod.apply_scale_factors(mod.dict['scale factors'], self.bm)
        assert any(self.bm.logTransformParams) and not all(self.bm.logTransformParams)
        assert self.bm.useScaleFactors
        assert self.bm.parametersBounded
        assert all(np.array(self.bm.lb) == np.array(lb))
        assert all(np.array(self.bm.ub) == np.array(ub))

    @pytest.mark.cheap
    def test_other_problems(self):
        # Basic check for another benchmarker
        mod = modification.Modification(logTransform='on', parameterBounds='on', scaleFactors='On')
        newbm = ionbench.problems.loewe2016.IKr()
        mod.apply(newbm)
        assert newbm.useScaleFactors
        assert all(np.array(newbm.logTransformParams) == np.array(newbm.STANDARD_LOG_TRANSFORM))
        assert any(newbm.logTransformParams)
        assert not all(newbm.logTransformParams)
        assert newbm.parametersBounded

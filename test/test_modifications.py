import ionbench
from ionbench import modification
import numpy as np
import pytest


class Test_Modifications:
    bm = ionbench.problems.staircase.HH()

    @pytest.mark.cheap
    def test_bounds(self):
        # Check bounds are correctly applied for all settings
        # No bounds initially
        assert not self.bm._bounded
        # Positive
        mod = modification.Modification(bounds='positive')
        mod.apply(self.bm)
        assert self.bm._bounded
        assert all(np.array(self.bm.lb) == 0)
        assert all(np.array(self.bm.ub) == np.inf)
        # Sampler
        mod = modification.Modification(bounds='Sampler')
        mod.apply(self.bm)
        assert self.bm._bounded
        assert all(np.array(self.bm.lb) > 0)
        assert all(np.array(self.bm.ub) < np.inf)
        # Custom
        lb = np.random.rand(self.bm.n_parameters())
        ub = np.random.rand(self.bm.n_parameters()) + 1
        mod = modification.Modification(bounds='Custom', customBounds=[lb, ub])
        mod.apply(self.bm)
        assert self.bm._bounded
        assert all(self.bm.lb == lb)
        assert all(self.bm.ub == ub)
        # None
        mod = modification.Modification()
        mod.apply(self.bm)
        assert not self.bm._bounded

    @pytest.mark.cheap
    def test_log_transform(self):
        # Check log transform is correctly applied for all settings
        # No log transform initially
        assert not any(self.bm._logTransformParams)
        # Standard log transform
        mod = modification.Modification(logTransform='standard')
        mod.apply(self.bm)
        assert any(self.bm._logTransformParams)
        assert not all(self.bm._logTransformParams)
        # Full log transform
        mod = modification.Modification(logTransform='Full')
        mod.apply(self.bm)
        assert all(self.bm._logTransformParams)
        # Custom
        transforms = list(np.random.choice([True, False], self.bm.n_parameters() - 2)) + [True, False]  # Random true and falses in which the last two are True and False
        mod = modification.Modification(logTransform='custom', customLogTransform=transforms)
        mod.apply(self.bm)
        assert any(self.bm._logTransformParams)
        assert all(np.array(self.bm._logTransformParams) == np.array(transforms))
        assert not all(self.bm._logTransformParams)
        # No log transform
        mod = modification.Modification()
        mod.apply(self.bm)
        assert not any(self.bm._logTransformParams)

    @pytest.mark.cheap
    def test_scale_factors(self):
        # Check scale factors are correctly applied
        # No scale factors initially
        assert not self.bm._useScaleFactors
        # Turn on scale factors
        mod = modification.Modification(scaleFactors='on')
        mod.apply(self.bm)
        assert self.bm._useScaleFactors
        # Default is off
        mod = modification.Modification()
        mod.apply(self.bm)
        assert not self.bm._useScaleFactors

    @pytest.mark.cheap
    def test_multiple_settings(self):
        # Check multiple settings applied at once doesn't break anything and order in the .apply() method doesn't matter
        mod = modification.Modification(logTransform='Full', bounds='positive', scaleFactors='On')
        mod.apply_log_transforms(mod.dict['log transform'], self.bm)
        mod.apply_scale_factors(mod.dict['scale factors'], self.bm)
        mod.apply_bounds(mod.dict['bounds'], self.bm)
        assert all(self.bm._logTransformParams)
        assert self.bm._useScaleFactors
        assert self.bm._bounded
        lb = self.bm.lb
        ub = self.bm.ub
        # Check order of applying doesnt change bounds
        emptyMod = modification.Empty()
        emptyMod.apply(self.bm)
        mod = modification.Modification(logTransform='Full', bounds='positive', scaleFactors='on')
        mod.apply_bounds(mod.dict['bounds'], self.bm)
        mod.apply_log_transforms(mod.dict['log transform'], self.bm)
        mod.apply_scale_factors(mod.dict['scale factors'], self.bm)
        assert all(self.bm._logTransformParams)
        assert self.bm._useScaleFactors
        assert self.bm._bounded
        assert all(np.array(self.bm.lb) == np.array(lb))
        assert all(np.array(self.bm.ub) == np.array(ub))

    @pytest.mark.cheap
    def test_other_problems(self):
        # Basic check for another benchmarker
        mod = modification.Modification(logTransform='standard', bounds='sampler', scaleFactors='On')
        newbm = ionbench.problems.loewe2016.IKr()
        mod.apply(newbm)
        assert newbm._useScaleFactors
        assert all(np.array(newbm._logTransformParams) == np.array(newbm.standardLogTransform))
        assert any(newbm._logTransformParams)
        assert not all(newbm._logTransformParams)
        assert newbm._bounded

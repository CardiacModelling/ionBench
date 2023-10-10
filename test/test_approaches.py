import pytest
import ionbench
from ionbench import approach
import numpy as np

class Test_Approaches:
    bm = ionbench.problems.staircase.HH_Benchmarker()
    def test_bounds(self):
        #No bounds initially
        assert not self.bm._bounded
        #Positive
        app = approach.Approach(bounds = 'positive')
        app.apply(self.bm)
        assert self.bm._bounded
        assert all(np.array(self.bm.lb)==0)
        assert all(np.array(self.bm.ub)==np.inf)
        #Sampler
        app = approach.Approach(bounds = 'Sampler')
        app.apply(self.bm)
        assert self.bm._bounded
        assert all(np.array(self.bm.lb)>0)
        assert all(np.array(self.bm.ub)<np.inf)
        #Custom
        lb = np.random.rand(self.bm.n_parameters())
        ub = np.random.rand(self.bm.n_parameters())+1
        app = approach.Approach(bounds = 'Custom', customBounds = [lb,ub])
        app.apply(self.bm)
        assert self.bm._bounded
        assert all(self.bm.lb==lb)
        assert all(self.bm.ub==ub)
        #None
        app = approach.Approach()
        app.apply(self.bm)
        assert not self.bm._bounded
    
    def test_log_transform(self):
        #No log transform initially
        assert not any(self.bm._logTransformParams)
        #Standard log transform
        app = approach.Approach(logTransform = 'standard')
        app.apply(self.bm)
        assert any(self.bm._logTransformParams)
        assert not all(self.bm._logTransformParams)
        #Full log transform
        app = approach.Approach(logTransform = 'Full')
        app.apply(self.bm)
        assert all(self.bm._logTransformParams)
        #Custom
        transforms = list(np.random.choice([True, False],self.bm.n_parameters()-2))+[True, False] #Random true and falses in which the last two are True and False
        app = approach.Approach(logTransform = 'custom', customLogTransform = transforms)
        app.apply(self.bm)
        assert any(self.bm._logTransformParams)
        assert all(np.array(self.bm._logTransformParams) == np.array(transforms))
        assert not all(self.bm._logTransformParams)
        #No log transform
        app = approach.Approach()
        app.apply(self.bm)
        assert not any(self.bm._logTransformParams)
    
    def test_scale_factors(self):
        #No scale factors initially
        assert not self.bm._useScaleFactors
        #Turn on scale factors
        app = approach.Approach(scaleFactors = 'on')
        app.apply(self.bm)
        assert self.bm._useScaleFactors
        #Default is off
        app = approach.Approach()
        app.apply(self.bm)
        assert not self.bm._useScaleFactors
    
    def test_multiple_settings(self):
        app = approach.Approach(logTransform='Full',bounds='positive',scaleFactors='On')
        app.apply_log_transforms(app.dict['log transform'], self.bm)
        app.apply_scale_factors(app.dict['scale factors'], self.bm)
        app.apply_bounds(app.dict['bounds'], self.bm)
        assert all(self.bm._logTransformParams)
        assert self.bm._useScaleFactors
        assert self.bm._bounded
        lb = self.bm.lb
        ub = self.bm.ub
        #Check order of applying doesnt change bounds
        emptyApp = approach.Empty()
        emptyApp.apply(self.bm)
        app = approach.Approach(logTransform='Full',bounds='positive',scaleFactors='on')
        app.apply_bounds(app.dict['bounds'], self.bm)
        app.apply_log_transforms(app.dict['log transform'], self.bm)
        app.apply_scale_factors(app.dict['scale factors'], self.bm)
        assert all(self.bm._logTransformParams)
        assert self.bm._useScaleFactors
        assert self.bm._bounded
        assert all(np.array(self.bm.lb) == np.array(lb))
        assert all(np.array(self.bm.ub) == np.array(ub))
    
    def test_other_problems(self):
        app = approach.Approach(logTransform = 'standard', bounds = 'sampler', scaleFactors = 'On')
        newbm = ionbench.problems.loewe2016.ikr()
        app.apply(newbm)
        assert newbm._useScaleFactors
        assert all(np.array(newbm._logTransformParams) == np.array(newbm.standardLogTransform))
        assert any(newbm._logTransformParams)
        assert not all(newbm._logTransformParams)
        assert newbm._bounded

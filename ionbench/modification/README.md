# Modifications & Modification Class
## Summary
The file *modification.py* contains the Modification class and its subclasses (the specific modifications used in the ionBench results). 

The Modification class stores modification settings that can be applied to a benchmarker problem. 

The subclasses of the Modification class are the specific modifications that can be applied to a benchmarker problem. They can be accessed using the modification names (same as the approach names in the paper, excluding the "a", "b", etc. suffixes, apart from Wilhelms2012a and Wilhelms2012b, which use different modifications).

The modifications can also be accessed by using the optimisers, where each optimiser has a `get_modification()` function that loads all modifications for the approaches that use that optimiser.

To access only the modifications of the unique approaches, use `ionbench.APP_UNIQUE`, explained [here](../../ionbench/README.md).

## Methods
### User facing methods
There are two methods relevant to a user of ionBench to use modifications. These are: `__init__()` and `apply()`.

#### Inputs into Modification()
When initialising a Modification (not relevant when loading the subclasses) with the call `ionbench.modification.Modification()`, it is necessary to specify the components of the modification. There are four settings that can be enabled, `logTransform`, `scaleFactors`, `parameterBounds`, and `rateBounds`. Each of these settings can be turned on by setting the input to `'on'` rather than the default of `'off'`. 

You can also use custom log transforms and custom bounds to set which parameters are transformed and the upper and lower bounds on each parameter, respectively.

Specifying a name for the modification can be done with `name='my modification name'`.

You can also specify additional keyword arguments for optimisers that can be automatically applied in the [multistart](../../ionbench/utils/multistart.py) function.

#### apply()
The apply method is used to apply the modification to a problem. The inputs to the apply function are the benchmarker problem to be modified.

The calls the four internal methods `apply_log_transforms()`, `apply_scale_factors()`, `apply_parameter_bounds()`, and `apply_rate_bounds()` in order to apply the modification to the problem.


### Internal methods
The four internal methods for the Modification class are: `apply_log_transforms()`, `apply_scale_factors()`, `apply_parameter_bounds()`, and `apply_rate_bounds()`. 

These take inputs of a setting (supplied by the `.apply()` method using `self.dict`) and a benchmarker problem. They use the benchmarker functions to apply the modification.

## Attributes
### User facing attributes
There are no attributes in the Modification class that need to be controlled by the user. 

### Internal attributes
Once the inputs to the Modification class have been set by the user, the information is stored in a dictionary (`self.dict`). Custom log transforms and parameter bounds are stored separately (`self.customLogTransform` and `self.customBounds`). Additional keyword arguments are stored as a dictionary (`self.kwargs`).

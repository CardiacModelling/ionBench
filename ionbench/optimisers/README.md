# Optimisers
## Summary
This directory contains all of the optimisers implemented into ionBench. The directory is split into three subdirectories, each containing different optimisers. The three subdirectories are:
* __external_optimisers__
* __scipy_optimisers__
* __pints_optimisers__
* contains implementations of optimisers for ionBench
* directory structure

## Optimiser interface
Each optimiser is implemented as a python function. This function takes an input of a benchmarker problem, and an optional input of starting parameters. These functions are stored inside modules, with each optimiser function being named `run()` inside a module corresponding to the optimiser. The optimiser function returns the best parameters found by the optimiser. 

Modifications for each optimiser can be gathered from the optimiser's module's `get_modification()` function. 

## How to define a new optimiser
If you wish to define a new optimiser to use with ionBench, it only needs a couple of requirements of the above interface. The optimiser must use a benchmarkers cost function `bm.cost()`. While it doesn't need to be a python function, it is recommended over a script, ideally including the benchmarker problem as an input.

## Scipy Optimisers
The [scipy optimisers](../../ionbench/optimisers/scipy_optimisers) directory contains the optimisers that are implemented directly using the ***scipy*** Python package.

Further details can be found in the corresponding [readme](../../ionbench/optimisers/scipy_optimisers/README.md).

## Pints Optimisers
The [pints optimisers](../../ionbench/optimisers/pints_optimisers) directory contains the optimisers that are implemented directly using the ***pints*** Python package. 

Further details can be found in the corresponding [readme](../../ionbench/optimisers/pints_optimisers/README.md).

## External Optimisers
The [external optimisers](../../ionbench/optimisers/external_optimisers) directory contains the remaining optimisers where implementations were not available in ***pints*** or ***scipy***. These have all been implemented specifically for ionBench.

Further details can be found in the corresponding [readme](../../ionbench/optimisers/external_optimisers/README.md).

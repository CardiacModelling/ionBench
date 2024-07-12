# ionBench
## Summary
This folder contains the code for the ionBench package. This package is a benchmarking tool for comparing different parameter optimization algorithms for ion channel models. The package is designed to allow for easy addition of new optimisation approaches.

## Folder structure
The folder structure for the ionBench package is outlined below. 
```
├───benchmarker
├───data
│   ├───loewe2016
│   ├───moreno2016
│   ├───staircase
│   └───test
├───modification
├───optimisers
│   ├───external_optimisers
│   ├───pints_optimisers
│   └───scipy_optimisers
├───problems
├───tracker
├───uncertainty
└───utils
```

* The [benchmarker](../ionbench/benchmarker) directory contains the main __Benchmarker__ class that the test problems all inherit from and defines the core features of the benchmarkers.

* The [data](../ionbench/data) directory is split up into the available benchmark problems. Each subdirectory contains the Myokit *.mmt* files, the voltage clamp protocols stored as *.csv* files where relevant, and output data to train the models, also stored as a *.csv*. It also contains the data for the [test](../ionbench/problems/test.py) problem which is used for testing ***ionBench***.

* The [modification](../ionbench/modification) directory contains the modification classes, generalised settings for handling transformations and bounds. 

* The [optimisers](../ionbench/optimisers) directory contains all the optimisation algorithms that are currently implemented. These are then divided into three subdirectories, containing the optimisers from ***pints***, from ***scipy***, and other optimisation algorithms used in fitting ion channel models that have been implemented specifically for ***ionBench***.

* The [problems](../ionbench/problems) directory contains the classes for the available benchmarking problems. This features the problems from Loewe et al. 2016 and Moreno et al. 2016. In addition to these previously defined problems, we have introduced two further problems, a Hodgkin-Huxley IKr model from Beattie et al. 2017 and a Markov IKr model from Fink et al. 2008 using the staircase protocol. 

* The [tracker](../ionbench/tracker) directory contains the __Tracker__ class which records the performance metrics over the course of an optimisation.

* The [uncertainty](../ionbench/uncertainty) directory contains functions for determining uncertainty and unidentifiability in the problems, such as calculating profile likelihood plots.

* The [utils](../ionbench/utils) directory contains utility functions for the operation of ionBench. This includes code for handling the steady states of the models and a function to initiate multiple runs of the same approach and record the results.

## \_\_init__.py
The [\_\_init__.py](../ionbench/__init__.py) file is run upon importing ***ionBench*** and defines a handful of variables used throughout the package. This includes the path to the [data](../ionbench/data) and [root](../ionbench) directories (note the root directory is the directory in which the ionBench package is installed, not the [root directory of the repo](..)). It also points to the optimisation approaches and optimisers available in ***ionBench***, explained below.

### Optimisers
`ionbench.OPT_SCIPY`, `ionbench.OPT_PINTS`, and `ionbench.OPT_EXT` are lists of the optimisers available in ***ionBench*** (lists of strings, pointing to the optimiser modules, split into [scipy optimisers](../ionbench/optimisers/scipy_optimisers), [pints optimisers](../ionbench/optimisers/pints_optimisers), and [external optimisers](../ionbench/optimisers/external_optimisers), respectively).

These can be used to easily access the optimisers available in ***ionBench***, for example:

```python
import ionbench
import importlib

bm = ionbench.problems.staircase.HH()
for optimiser in ionbench.OPT_SCIPY:
    module = importlib.import_module(optimiser)
    x = module.run(bm)
    print(x)
    bm.reset()
```

`ionbench.OPT_ALL` is the concatenation of these lists.

### Modifications
We also define `ionbench.N_MOD_SCIPY`, `ionbench.N_MOD_PINTS`, and `ionbench.N_MOD_EXT` which are the number of modifications used for each optimiser, in the order matching `ionbench.OPT_SCIPY`, `ionbench.OPT_PINTS`, and `ionbench.OPT_EXT`. These can be used to easily access the modifications available in ***ionBench*** through the `get_modification()` function.

We also have `ionbench.N_MOD_ALL` to match `ionbench.OPT_ALL`.

```python
import ionbench
import importlib

bm = ionbench.problems.staircase.HH()
for i, optimiser in enumerate(ionbench.OPT_SCIPY):
    module = importlib.import_module(optimiser)
    for modNum in range(ionbench.N_MOD_SCIPY[i]):
        modification = module.get_modification(modNum)
        modification.apply(bm)
        x = module.run(bm)
        print(x)
        bm.reset()
```

### Approaches
To streamline this process, we also define the approaches in ***ionBench***. `ionbench.APP_SCIPY`, `ionbench.APP_PINTS`, and `ionbench.APP_EXT` are lists of dictionaries, each containing a `module` item pointing to the optimiser module, a `modNum` item pointing to the modification number, and an option to specify additional `kwargs` that can be passed into the optimiser `.run()` function by [multistart](../ionbench/utils/multistart.py).

We also define `ionbench.APP_ALL` and `ionbench.APP_UNIQUE` which are the concatenation of these lists and the unique approaches, respectively.

```python
import ionbench
import importlib

bm = ionbench.problems.staircase.HH()
for optimiser in ionbench.OPT_SCIPY:
    module = importlib.import_module(optimiser)
    x = module.run(bm)
    print(x)
    bm.reset()
```

### Caching
Finally, we have a global (for ***ionBench***) variable, `ionbench.cache_enabled` (defaults to `True`) which can be set to `False` to disable caching of the model outputs. See [here](../ionbench/utils) for more information.

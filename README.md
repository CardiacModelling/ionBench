# ionBench
A benchmarking tool for comparing different parameter optimization algorithms for ion channel models.

## Project Structure
The tree structure of this project is outlined below. 
```
├───docs
├───ionbench
├───scipts
│   ├───text
│   └───figures
└───test
```

The [docs](docs) directory contains information and guides on how to use the benchmarker problems and the optimisation algorithms. 

The [ionbench](ionbench) directory contains the majority of the code, including the benchmarker and problems classes and the different optimisation algorithms. This is what is installed using pip.

The [scripts](scripts) directory contains scripts for generating figures, tables and data for the paper.

The [test](test) directory contains unit tests for ensuring changes do not break previous functionality.

## Installation
### Installing ionBench package
The ***ionBench*** package can be installed from PyPI using pip.

```pip install ionbench```

Note that ***ionBench*** uses ***myokit*** to do its simulations, which relies on CVODES (from Sundials). For Linux and macOS users a working installation of CVODES is required. For Windows users, CVODES should be automatically installed with ***myokit***.

### Getting the scripts and results
The scripts are results from ***ionBench*** are available in this repo. If you also want access to these, you should:

1. Clone the repository.
2. Install ***ionBench*** locally using pip (navigate to the the root directory of the repo and run ```pip install -e .```) or install ***ionBench*** from PyPI (```pip install ionbench```).
3. Navigate to the scripts directory and run the scripts you are interested in.

## Getting Started
If you want to use ***ionBench***, check out the [introduction.md](docs/introduction.md) and [tutorial.ipynb](docs/tutorial.ipynb) in the [docs](docs) directory.

## Workflow
The intended workflow for using the benchmarker is to generate a benchmarker object, set up the optimisers modification and apply it to the benchmarker, and pass the benchmarker into the optimisation algorithm to evaluate. All optimisers should accept a single benchmarker as input with all other inputs being optional. 

```python
import ionbench

bm = ionbench.problems.staircase.HH()
modification = ionbench.optimisers.pints_optimisers.cmaes_pints.get_modification()
modification.apply(bm)
optimisedParameters = ionbench.optimisers.scipy_optimisers.nelderMead_scipy.run(bm)
```

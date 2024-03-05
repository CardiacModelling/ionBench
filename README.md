# ionBench
A benchmarking tool for comparing different parameter optimization algorithms for ion channel models.

## Project Structure
The tree structure of this project is outlined below. 
```
├───docs
├───ionbench
│   ├───benchmarker
│   ├───data
│   │   ├───loewe2016
│   │   ├───moreno2016
│   │   ├───staircase
│   │   └───test
│   ├───modification
│   ├───optimisers
│   │   ├───external_optimisers
│   │   ├───pints_optimisers
│   │   └───scipy_optimisers
│   ├───problems
│   ├───tracker
│   ├───uncertainty
│   └───utils
├───scipts
│   └───figures
└───test
```

The __docs__ directory contains information and guides on how to use the benchmarker problems and the optimisation algorithms. 

The __ionbench__ directory contains the majority of the code, including the benchmarker and problems classes and the different optimisation algorithms. This is what is installed using pip.

* The __benchmarker__ subdirectory contains the main __Benchmarker__ class that the test problems all inherit from and defines the core features of the benchmarkers.

* The __data__ subdirectory is split up into the available benchmark problems. Each subdirectory contains the Myokit *.mmt* files, the voltage clamp protocols stored as *.csv* files where relevant, and output data to train the models, also stored as a *.csv*. It also contains the data for the *test* problem which is used for testing the benchmarker.

* The __modification__ subdirectory contains the modification classes, generalised settings for handling transformations and bounds. 

* The __optimisers__ subdirectory contains all the optimisation algorithms that are currently implemented. These are then further subdivided into three directories, containing the optimisers from ***pints***, from ***scipy***, and other optimisation algorithms used in fitting ion channel models that have been implemented specifically for ***ionBench***.

* The __problems__ subdirectory contains the classes for the available benchmarking problems. This features the problems from Loewe et al. 2016 and Moreno et al. 2016. In addition to these previously defined problems, we have introduced two further problems, a Hodgkin-Huxley IKr model from Beattie et al. 2017 and a Markov IKr model from Fink et al. 2008 using the staircase protocol. 

* The __tracker__ subdirectory contains the __Tracker__ class which records the performance metrics over the course of an optimisation.

* The __uncertainty__ subdirectory contains functions for determining uncertainty and unidentifiability in the problems, such as calculating profile likelihood plots and Fisher's Information Matrix.

* The __utils__ subdirectory contains utility functions for the operation of ionBench. This includes code for handling the steady states of the models and a function to initiate multiple runs of the same approach and record the results.

The __scripts__ directory contains scripts for generating figures, tables and data for the paper.

The __test__ directory contains unit tests for ensuring changes do not break previous functionality.

## Installation
***ionBench*** can be installed using pip.

```pip install ionbench```

Note that ***ionBench*** uses ***myokit*** to do its simulations, which relies on CVODES (from Sundials). For Linux and macOS users a working installation of CVODES is required. For Windows users, CVODES should be automatically installed with ***myokit***.

## Getting Started
If you want to use ***ionBench***, check out the __introduction__ and __tutorial__ in the __docs__ directory.

## Workflow
The intended workflow for using the benchmarker is to generate a benchmarker object, set up the optimisers modification and apply it to the benchmarker, and pass the benchmarker into the optimisation algorithm to evaluate. All optimisers should accept a single benchmarker as input with all other inputs being optional. 
```
import ionbench
bm = ionbench.problems.staircase.HH()
modification = ionbench.optimisers.pints_optimisers.cmaes_pints.get_modification()
modification.apply(bm)
optimisedParameters = ionbench.optimisers.scipy_optimisers.nelderMead_scipy.run(bm)
```

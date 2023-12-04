# ionBench
A benchmarking tool for comparing different parameter optimization algorithms for ion channel models.

## Project Structure
The tree structure of this project is outlined below. 
```
├───docs
├───ionbench
│   ├───modification
│   ├───benchmarker
│   ├───data
│   │   ├───loewe2016
│   │   ├───moreno2016
│   │   └───staircase
│   ├───optimisers
│   │   ├───external_optimisers
│   │   ├───pints_optimisers
│   │   ├───scipy_optimisers
│   │   └───spsa_spsa.py
│   ├───problems
│   ├───uncertainty
│   └───multistart.py
└───test
```

The __docs__ directory contains information and guides on how to use the benchmarker, the test problems and the optimisation algorithms. 

The __ionbench__ directory contains the majority of the code, including the benchmarker and problems classes and the different optimisation algorithms. This is what is installed using pip.

* The __modification__ subdirectory contains the modification classes, generalised settings for handling transformations and bounds. 

* The __benchmarker__ subdirectory contains the main __Benchmarker__ class that the test problems all inherit from and defines the core features of the benchmarkers. It also contains the __Tracker__ class which contains the functions to take performance metrics over time.

* The __data__ subdirectoy is split up into the available test problems. Each subdirectory contains the Myokit *.mmt* files, the voltage clamp protocols stored as *.csv* files where relevant, and output data to train the models, also stored as a *.csv*.

* The __optimisers__ subdirectory contains all of the optimisation algorithms that are currently implemented. These are then further subdivided into three directories, containing the optimisers from ***pints***, from ***scipy***, and other optimisation algorithms used in fitting ion channel models that have been implemented specifically for ***ionBench***.

* The __problems__ subdirectory contains the classes for the available benchmarking problems. This features the problems from Loewe et al 2016 and Moreno et al 2016. In addition to these previously defined problems, we have introduced two further test problems, a Hodgkin-Huxley IKr model from Beattie et al 2017 and a Markov IKr model from Fink et al 2008. 

* The final subdirectory, __uncertainty__, contains functions for determining uncertainty and unidentifiability in the problems, such as calculating profile likelihood plots and Fisher's Information Matrix.

* __multistart.py__ provides a tool for rerunning an optimiser to derive average performace metrics.

The __test__ directory contains unit tests for ensuring changes do not break previous functionality.

## Installation
***ionBench*** can be installed using pip.

```pip install ionbench```

Note that ***ionBench*** uses ***myokit*** to do its simulations, which relies on CVODES (from Sundials). For Linux and Mac OS users a working installation of CVODES is required. For Windows users, CVODES should be automatically installed with ***myokit***.

## Getting Started
If you want to use ***ionBench***, check out the __introduction__ and __tutorial__ in the __docs__ directory.

## Workflow
The intended workflow for using the benchmarker is to generate a benchmarker object, setup the optimisers modification and apply it to the benchmarker, and pass the benchmarker into the optimisation algorithm to evaluate. All optimisers should accept a single benchmarker as input with all other inputs being optional. 
```
import ionbench
bm = ionbench.problems.staircase.HH_Benchmarker()
modification = ionbench.optimisers.pints_optimisers.cmaes_pints.get_modification()
modification.apply(bm)
optimisedParameters = ionbench.optimisers.scipy_optimisers.nelderMead_scipy.run(bm)
```

## Future Features
* Bounds - Current only bounds on the parameters can be included in the benchmarker but it would be nice to have bounds on the rates. Additionally, it would be good to include barrier function style bounds to allow them to work nicely with gradient based methods.

* Parallelisation - Its not clear yet how well the benchmarker would handle being run in parallel (specifically for the tracker) but it is something that would be worth looking into.

* Real data - Both Moreno et al 2016 and Loewe et al 2016 include real data in the papers. It would be nice to see how the algorithms handle fitting to real data but its not clear how to best implement the performance metrics, two of which rely on knowing the true parameters.

# Pints Optimisers
In ionBench, we have implementations of some optimisers from ***pints***. These optmisers use ionbench.utils.classes_pints to interface between ***pints*** and ***ionBench***.

A summary of ionbench.utils.classes_pints is given in the corresponding [README.md](../../../ionbench/utils) file.

## CMA-ES
Covariance Matrix Adaptation Evolution Strategy (CMA-ES) is a gradient free optimiser available in ***pints***. It is the only optimiser used in the ionBench results, being used in two unique approaches. These approaches also differ in hyperparameters.

## Nelder Mead
Nelder Mead is a simplex optimiser available in ***pints***. While Nelder Mead is used in some approaches, we instead use the [scipy](../../../ionbench/optimisers/scipy_optimisers) implementation.

## PSO
Particle Swarm Optimisation (PSO) is available in ***pints***. This PSO implementation is not used in any approaches. 

## rProp
This uses IRPropMin from ***pints***. This is not used in any approaches. 

## SNES
Separable Natural Evolution Strategy (SNES) is a gradient free optimiser available in ***pints***. This is not used in any approaches.

## XNES
Expotential Natural Evolution Strategy (XNES) is a gradient free optimiser available in ***pints***. This is not used in any approaches.

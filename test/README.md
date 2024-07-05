# Tests
## Summary
This directory contains the unit tests for ionBench. The tests are written using the `pytest` module in Python. 

## Problems
A large set of tests verifying many aspects of the problems, benchmarker, and tracker classes. These tests include tracking handling of repeated parameters and out of bounds points, verifying the gradient calculation, and parameter transforms.

## Modifications
Checks that the modifications apply the appropriate settings for newly constructed modifications, then verifies all modifications can be loaded from optimiser modules.

## Optimisers
The optimisers are tested against a simple [test problem](../ionbench/problems). The tests check that the optimisers are able to minimise this easy problem. We also verify that all optimisers can correctly trigger the maxIter flag.

## Multistart
The multistart test verifies that when multistart is called, it optimises from each of the specified input parameters. This is a simple test, mainly to ensure the code doesn't throw any errors.

## Uncertainty
The uncertainty tests verify that the uncertainty calculations and plotting run without errors. They do not make any attempt to ensure the calculations are correct.

## Coverage
Finally, we have some tests designed to increase the coverage of the codebase. These tests are mainly ones that are difficult to trigger in the normal course of running the code, typically requiring long optimisation runs.

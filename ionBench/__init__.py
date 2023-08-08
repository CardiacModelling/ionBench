import os
from ionBench import benchmarker
from ionBench import problems
from ionBench import scipy_optimisers
from ionBench import pints_optimisers
from ionBench import external_optimisers
from ionBench import spsa_spsa

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(os.path.dirname(ROOT_DIR),"data")

TEST_DIR = os.path.join(os.path.dirname(ROOT_DIR),"test")
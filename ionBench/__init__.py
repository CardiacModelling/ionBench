import os
from ionBench import benchmarker
from ionBench import problems
from ionBench import optimisers

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(os.path.dirname(ROOT_DIR),"data")

TEST_DIR = os.path.join(os.path.dirname(ROOT_DIR),"test")
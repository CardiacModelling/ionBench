import os
from ionbench import benchmarker
from ionbench import problems
from ionbench import optimisers
from ionbench import modification
from ionbench import uncertainty
from ionbench.multistart import *

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(ROOT_DIR,"data")

TEST_DIR = os.path.join(os.path.dirname(ROOT_DIR),"test")
import os
from ionbench import benchmarker
from ionbench import problems
from ionbench import optimisers
from ionbench import modification
from ionbench import uncertainty
from ionbench import utils

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(ROOT_DIR, "data")

OPT_SCIPY = ['ionbench.optimisers.scipy_optimisers.' + st + '_scipy' for st in ['conjugateGD', 'lm', 'nelderMead', 'powell', 'slsqp', 'trustRegionReflective']]

OPT_PINTS = ['ionbench.optimisers.pints_optimisers.' + st + '_pints' for st in ['cmaes', 'nelderMead', 'pso', 'snes', 'xnes', 'rprop']]

OPT_EXT = ['ionbench.optimisers.external_optimisers.' + st for st in ['curvilinearGD_Dokos2004', 'DE_Zhou2009', 'GA_Bot2012', 'GA_Cairns2017', 'GA_Gurkiewicz2007a', 'GA_Gurkiewicz2007b', 'GA_Smirnov2020', 'hybridPSOTRR_Loewe2016', 'hybridPSOTRRTRR_Loewe2016', 'NMPSO_Clausen2020', 'NMPSO_Liu2011', 'patternSearch_Kohjitani2022', 'ppso_Chen2012', 'pso_Cabo2022', 'PSO_Loewe2016', 'pso_Seemann2009', 'PSOTRR_Loewe2016', 'randomSearch_Vanier1999', 'SA_Vanier1999', 'SPSA_Spall1998', 'stochasticSearch_Vanier1999']]

OPT_ALL = OPT_SCIPY + OPT_PINTS + OPT_EXT

N_MOD_SCIPY = [2, 2, 3, 3, 1, 3]
N_MOD_PINTS = [3, 0, 0, 0, 0, 0]
N_MOD_EXT = [3, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
N_MOD_ALL = N_MOD_SCIPY + N_MOD_PINTS + N_MOD_EXT

APP_SCIPY = []
for i in range(len(OPT_SCIPY)):
    for j in range(N_MOD_SCIPY[i]):
        APP_SCIPY.append({'module': OPT_SCIPY[i], 'modNum': j + 1, 'kwargs': {}})

APP_PINTS = []
for i in range(len(OPT_PINTS)):
    for j in range(N_MOD_PINTS[i]):
        APP_PINTS.append({'module': OPT_PINTS[i], 'modNum': j + 1, 'kwargs': {}})

APP_EXT = []
for i in range(len(OPT_EXT)):
    for j in range(N_MOD_EXT[i]):
        APP_EXT.append({'module': OPT_EXT[i], 'modNum': j + 1, 'kwargs': {}})

APP_ALL = APP_SCIPY + APP_PINTS + APP_EXT
APP_UNIQUE = [APP_ALL[i] for i in range(len(APP_ALL)) if i not in [3, 8, 9, 12, 13, 16, 18, 19]]

cache_enabled = True

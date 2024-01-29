import ionbench
import numpy as np
import importlib

bm = ionbench.problems.staircase.MM_Benchmarker()
for app in ionbench.APP_SCIPY:
    try:
        bm.tracker.load(f"{app['module']}modNum{app['modNum']}_run4.pickle")
    except Exception as e:
        print(e)
        print(f"Tracking of {app['module']} failed.")
        continue
    for runNum in range(5):
        bm.tracker.load(f"{app['module']}modNum{app['modNum']}_run{runNum}.pickle")
        evals = bm.tracker.evals
        for i in range(len(evals)):
            p1, st1 = evals[i]
            for j in range(i):
                p2, st2 = evals[j]
                if all(p1 == p2):
                    if not (st1=='grad' and st2=='cost'):
                        print(f"Approach: {app['module']} + modNum: {app['modNum']}. Run: {runNum}. Solve Type 1 (later solve): {st1}, Solve Type 2 (early solve): {st2}")

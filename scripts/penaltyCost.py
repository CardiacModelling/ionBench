# Checks that all the problems show costs well below 1e5 (the minimum penalty cost).
# Maximum observed cost is ~5000 for Loewe IKur so penalty of 1e5 is fine.

import ionbench


def maximum_cost(bm):
    maxCost = 0
    maxP = None
    for _ in range(1000):
        p = bm.sample()
        cost = bm.cost(p)
        if maxCost < cost:
            maxCost = cost
            maxP = p
    print(f'Finished {bm.NAME}. Max cost: {maxCost}, at parameters: {maxP}')


bm = ionbench.problems.staircase.HH()
maximum_cost(bm)
bm = ionbench.problems.staircase.MM()
maximum_cost(bm)
bm = ionbench.problems.loewe2016.IKr()
maximum_cost(bm)
bm = ionbench.problems.loewe2016.IKur()
maximum_cost(bm)
bm = ionbench.problems.moreno2016.INa()
maximum_cost(bm)

import ionbench
import numpy as np
import csv

bm = ionbench.problems.staircase.HH_Benchmarker()
with open('resultsFile.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    titles = ['Optimiser Name', 'Mod Num']
    variables = ['Conv Cost', 'Best Cost', 'Cost FE', 'Grad FE', 'Parameters Identified', 'Average Cost FE Time', 'Average Grad FE Time']
    for i in range(5):
        for j in variables:
            titles.append(f'Run {i} - {j}')
    titles += ['Success Rate', 'Expected Cost FE', 'Expected Grad FE', 'Tier', 'Tier Score']
    writer.writerow(titles)
    for app in ionbench.APP_ALL:
        print(app['module'])
        costAtConv = []
        bestCost = []
        costEvalsAtConv = []
        gradEvalsAtConv = []
        paramIdenAtConv = []
        costAverTime = []
        gradAverTime = []
        try:
            bm.tracker.load(f"{app['module']}modNum{app['modNum']}_run0.pickle")
        except Exception as e:
            print(e)
            print(f"Tracking of {app['module']} failed. Are you sure that the .pickle file of results exists?")
            optimiserName = app['module'].split('.')[-1]
            modNum = app['modNum']
            row = [optimiserName, modNum]
            row += [np.nan]*(len(titles)-2)
            writer.writerow(row)
            continue
        for runNum in range(5):
            bm.tracker.load(f"{app['module']}modNum{app['modNum']}_run{runNum}.pickle")
            costAverTime.append(np.mean(bm.tracker.costTimes))
            gradAverTime.append(np.mean(bm.tracker.gradTimes))
            bestCost.append(bm.tracker.bestCost)
            finalParamId = bm.tracker.paramIdentifiedCount[-1]
            paramIdenAtConv.append(finalParamId)
            ifEqualFinalParamId = bm.tracker.paramIdentifiedCount == finalParamId
            ind = [i for i, x in enumerate(ifEqualFinalParamId) if x]  # Indexes where number of parameters identified is equal to the final count
            for i in ind:
                if all(ifEqualFinalParamId[i:]):
                    # All future points remain with this many parameters identified, therefore it is considered converged
                    costEvalsAtConv.append(bm.tracker.modelSolves[i])
                    gradEvalsAtConv.append(bm.tracker.gradSolves[i])
                    costAtConv.append(bm.tracker.costs[i])
                    break
        
            bm.reset()
        successRate = np.mean(np.array(paramIdenAtConv)==bm.n_parameters())
        if successRate == 1:
            expectedCostEvals = np.mean(costEvalsAtConv)
            expectedGradEvals = np.mean(gradEvalsAtConv)
            tier = 1
            score = expectedCostEvals + expectedGradEvals*gradAverTime/costAverTime
        elif successRate > 0:
            Tsucc = np.mean([costEvalsAtConv[t] for t in range(5) if paramIdenAtConv[t]==bm.n_parameters()])
            Tfail = np.mean([costEvalsAtConv[t] for t in range(5) if paramIdenAtConv[t]!=bm.n_parameters()])
            expectedCostEvals = Tsucc + Tfail*(1-successRate)/successRate
            Tsucc = np.mean([gradEvalsAtConv[t] for t in range(5) if paramIdenAtConv[t]==bm.n_parameters()])
            Tfail = np.mean([gradEvalsAtConv[t] for t in range(5) if paramIdenAtConv[t]!=bm.n_parameters()])
            expectedGradEvals = Tsucc + Tfail*(1-successRate)/successRate
            tier = 2
            score = expectedCostEvals + expectedGradEvals*gradAverTime/costAverTime
        else:
            tier = 3
            expectedCostEvals = np.inf
            expectedGradEvals = np.inf
            score = np.mean(costEvalsAtConv+gradEvalsAtConv*gradAverTime/costAverTime)/np.mean(paramIdenAtConv)
        optimiserName = app['module'].split('.')[-1]
        modNum = app['modNum']
        row = [optimiserName, modNum]
        for i in range(5):
            row += [costAtConv[i], bestCost[i], costEvalsAtConv[i], gradEvalsAtConv[i], paramIdenAtConv[i], costAverTime[i], gradAverTime[i]]
        row += [successRate, expectedCostEvals, expectedGradEvals, tier, score]
        writer.writerow(row)

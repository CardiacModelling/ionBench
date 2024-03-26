import ionbench
from ionbench.uncertainty.profile_likelihood import plot_profile_likelihood

debug = False

bm = ionbench.problems.staircase.HH()
plot_profile_likelihood(modelType='hh', numberToPlot=bm.n_parameters(), debug=debug)

bm = ionbench.problems.staircase.MM()
plot_profile_likelihood(modelType='mm', numberToPlot=bm.n_parameters(), debug=debug)

bm = ionbench.problems.loewe2016.IKr()
plot_profile_likelihood(modelType='ikr', numberToPlot=bm.n_parameters(), debug=debug)

bm = ionbench.problems.loewe2016.IKur()
plot_profile_likelihood(modelType='ikur', numberToPlot=bm.n_parameters(), debug=debug)

bm = ionbench.problems.moreno2016.INa()
plot_profile_likelihood(modelType='ina', numberToPlot=bm.n_parameters(), debug=debug)

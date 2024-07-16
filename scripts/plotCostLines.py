"""
Plot a slice of the cost function between the sampled points that begin the optimisation and the true parameters.
"""
import ionbench
import pints.plot
import numpy as np
import matplotlib.pyplot as plt
import os


# Load all benchmarkers
bms = [ionbench.problems.staircase.HH(), ionbench.problems.staircase.MM(), ionbench.problems.loewe2016.IKr(), ionbench.problems.moreno2016.INa()]
names = ['HH', 'MM', 'IKr', 'INa']

for bm, name in zip(bms, names):
    # Some parameters in the staircase problems are sampled, and best viewed, in a log space
    if 'staircase' in bm.NAME:
        bm.log_transform(bm.STANDARD_LOG_TRANSFORM)
    # Set the seed to the same as for the optimisation
    np.random.seed(0)
    x0 = bm.sample(10)
    # Construct a pints forward model
    model = ionbench.utils.classes_pints.Model(bm)
    # Set up the time vector
    if 'moreno' in bm.NAME:
        times = np.arange(len(bm.DATA))
    else:
        times = np.arange(0, bm.T_MAX, bm.TIMESTEP)
    # Create a pints error measure
    problem = pints.SingleOutputProblem(model, times, model.bm.DATA)
    error = pints.RootMeanSquaredError(problem)

    for i, p in enumerate(x0):
        # Plot cost function between the two points
        pints.plot.function_between_points(error, p, bm.input_parameter_space(bm._TRUE_PARAMETERS), evaluations=500)
        # Restrict y limits if penalty costs appear in the plot
        ax = plt.gca()
        line = ax.lines[0]
        if np.any(line.get_ydata() > 1e5):
            plt.ylim(0.95*np.min(line.get_ydata()), 1.05*np.max(line.get_ydata(), initial=0, where=line.get_ydata() < 1e5))
        plt.title(f'{name} cost function. Run {i}')
        plt.savefig(os.path.join(ionbench.ROOT_DIR, '..', 'scripts', 'figures', 'costPlots', f'{name.lower()}_{i}.png'))
        plt.show()

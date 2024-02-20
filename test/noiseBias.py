# %% Exploring noise bias in currently generated data
import ionbench
import numpy as np
import matplotlib.pyplot as plt

bm = ionbench.problems.staircase.MM()
bm._useScaleFactors = True
bm.plotter = False
data = bm.data
noiseLevel = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1]
nPoints = 10
params = []  # Full optimised parameter vectors
plotPointsX = []  # X coordinate of plot points, noise level
plotPointsC = []  # Colour of plot points, better cost than default params
p = np.ones(bm.n_parameters())
for n in noiseLevel:
    for j in range(nPoints):
        bm.data = data + np.random.normal(0, np.mean(np.abs(data)) * 0.05 * n, len(data))
        out = ionbench.optimisers.scipy_optimisers.lm_scipy.run(bm, x0=p)
        plotPointsX.append(n * 100)
        plotPointsC.append(bm.cost(out) < bm.cost(bm.input_parameter_space(bm.defaultParams)))
        params.append(out)
        if n == 0:
            break
    print(f'Noise level {n * 100}% complete')
ymax = np.max(params) * 1.05
ymin = np.min(params) * 1.05 if np.min(params) < 0 else 0
for i in range(bm.n_parameters()):
    plotPointsY = [p[i] for p in params]
    plt.figure()
    plt.plot([0, 100], [0.95, 0.95], linestyle=':', color='r', linewidth=1)
    plt.plot([0, 100], [1.05, 1.05], linestyle=':', color='r', linewidth=1)
    plt.plot([0, 100], [1, 1], color='k', linewidth=1)
    plt.scatter(plotPointsX, plotPointsY, c=['blue' if k else 'red' for k in plotPointsC], marker='.')
    plt.title('Noise bias - Markov Model')
    plt.ylabel(f'Optimised Value of parameter {i}')
    plt.xlabel('Noise Strength (%)')
    plt.ylim((ymin, ymax))
    plt.show()

# %% Noise bias as a function of data frequency
bm = ionbench.problems.staircase.MM()
bm._useScaleFactors = True
bm.plotter = False
# Cant use simulate to get data as current data is different length to new data. Will need to use solve_model and add in the extra bits
bm.freq = 0.01
bm.sim.reset()
bm.set_params(bm.defaultParams)
bm.set_steady_state(bm.defaultParams)
data = bm.solve_model(np.arange(0, bm.tmax, bm.freq))
# noiseLevel = [0, 0.02, 0.1, 0.5, 1]
noiseLevel = [0, 0.1, 1]
nPoints = 10
params = []  # Full optimised parameter vectors
plotPointsX = []  # X coordinate of plot points, noise level
plotPointsC = []  # Colour of plot points, better cost than default params
p = np.ones(bm.n_parameters())
for n in noiseLevel:
    for j in range(nPoints):
        bm.data = data + np.random.normal(0, np.mean(np.abs(data)) * 0.05 * n, len(data))
        out = ionbench.optimisers.scipy_optimisers.lm_scipy.run(bm, x0=p)
        plotPointsX.append(n * 100)
        plotPointsC.append(bm.cost(out) < bm.cost(bm.input_parameter_space(bm.defaultParams)))
        params.append(out)
        if n == 0:
            break
    print(f'Noise level {n * 100}% complete')
ymax = np.max(params) * 1.05
ymin = np.min(params) * 1.05 if np.min(params) < 0 else 0
for i in range(bm.n_parameters()):
    plotPointsY = [p[i] for p in params]
    plt.figure()
    plt.plot([0, 100], [0.95, 0.95], linestyle=':', color='r', linewidth=1)
    plt.plot([0, 100], [1.05, 1.05], linestyle=':', color='r', linewidth=1)
    plt.plot([0, 100], [1, 1], color='k', linewidth=1)
    plt.scatter(plotPointsX, plotPointsY, c=['blue' if k else 'red' for k in plotPointsC], marker='.')
    plt.title(f'Effect of {1 / bm.freq}kHz data on noise bias - Markov Model')
    plt.ylabel(f'Optimised Value of parameter {i}')
    plt.xlabel('Noise Strength (%)')
    plt.ylim((ymin, ymax))
    plt.show()

# %% What does noise bias actually change in the current
bm = ionbench.problems.staircase.MM()
bm._useScaleFactors = True
bm.plotter = False
bm.freq = 0.1
bm.sim.reset()
bm.set_params(bm.defaultParams)
bm.set_steady_state(bm.defaultParams)
data = bm.solve_model(np.arange(0, bm.tmax, bm.freq))
p = np.ones(bm.n_parameters())
bm.data = data + np.random.normal(0, np.mean(np.abs(data)) * 0.05, len(data))
out = ionbench.optimisers.scipy_optimisers.lm_scipy.run(bm, x0=p)

optCurrent = bm.simulate(out, np.arange(0, bm.tmax, bm.freq))  # Noise optimised current
defCurrent = bm.simulate(p, np.arange(0, bm.tmax, bm.freq))  # Default current
plt.figure()
plt.title('Difference between optimised and default current')
plt.xlabel('Time (ms)')
plt.ylabel('Current Difference')
plt.plot(np.arange(0, bm.tmax, bm.freq), optCurrent - defCurrent)
plt.show()

# %% Sensitivity plots
bm = ionbench.problems.staircase.MM(sensitivities=True)
p = bm.defaultParams
# Get sensitivities
bm.simSens.reset()
bm.set_params(p)
bm.set_steady_state(p)

log, e = bm.simSens.run(bm.tmax + 1, log_times=np.arange(0, bm.tmax, bm.freq))

curr, sens = np.array(log[bm._outputName]), e
sens = np.array(sens)

plt.figure(figsize=(16, 9), dpi=300)
for i in range(bm.n_parameters()):
    if i in [0, 1, 7]:
        plt.plot(np.arange(0, bm.tmax, bm.freq), sens[:, 0, i], label=f'p{i}', linestyle=':')
    elif i in [2, 3, 4, 8, 9]:
        plt.plot(np.arange(0, bm.tmax, bm.freq), sens[:, 0, i], label=f'p{i}', linestyle='--')
    else:
        plt.plot(np.arange(0, bm.tmax, bm.freq), sens[:, 0, i], label=f'p{i}')
plt.title('Sensitivities')
plt.xlabel('Time (ms)')
plt.ylabel('dIKr/dpi(t)')
plt.legend(ncols=7)
plt.show()

plt.figure(figsize=(16, 9), dpi=300)
for i in range(bm.n_parameters()):
    if i in [0, 1, 7]:
        plt.plot(np.arange(0, bm.tmax, bm.freq), p[i] * sens[:, 0, i], label=f'p{i}', linestyle=':', zorder=3)
    elif i in [2, 3, 4, 8, 9]:
        plt.plot(np.arange(0, bm.tmax, bm.freq), p[i] * sens[:, 0, i], label=f'p{i}', linestyle='--', zorder=2)
    else:
        plt.plot(np.arange(0, bm.tmax, bm.freq), p[i] * sens[:, 0, i], label=f'p{i}', zorder=1)
plt.title('Scaled sensitivities')
plt.xlabel('Time (ms)')
plt.ylabel('pi*dIKr/dpi(t)')
plt.legend(ncols=7)
plt.show()

plt.figure(figsize=(16, 9), dpi=300)
for i in range(bm.n_parameters()):
    if i in [0, 1, 7]:
        plt.plot(np.arange(0, bm.tmax, bm.freq), np.abs(p[i] * sens[:, 0, i]), label=f'p{i}', linestyle=':', zorder=3)
    elif i in [2, 3, 4, 8, 9]:
        plt.plot(np.arange(0, bm.tmax, bm.freq), np.abs(p[i] * sens[:, 0, i]), label=f'p{i}', linestyle='--', zorder=2)
    else:
        plt.plot(np.arange(0, bm.tmax, bm.freq), np.abs(p[i] * sens[:, 0, i]), label=f'p{i}', zorder=1)
plt.title('Scaled sensitivity magnitudes')
plt.xlabel('Time (ms)')
plt.ylabel('|pi*dIKr/dpi(t)|')
plt.legend(ncols=7)
plt.show()

plt.figure(figsize=(16, 9), dpi=300)
for i in range(bm.n_parameters()):
    plt.plot(np.arange(0, bm.tmax, bm.freq), log['Environment.V'])
plt.title('Staircase Protocol')
plt.xlabel('Time (ms)')
plt.ylabel('Voltage (mV)')
plt.show()

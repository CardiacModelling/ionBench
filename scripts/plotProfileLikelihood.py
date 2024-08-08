import ionbench
from ionbench.uncertainty.profile_likelihood import plot_profile_likelihood
import os

# Set working directory to results locations
script_path = os.path.abspath(os.path.dirname(__file__))
results_dir = os.path.join(script_path, '..', 'results', 'uncertainty')
os.chdir(results_dir)

debug = False
filepath = os.path.join(os.getcwd(), '..', '..', 'scripts', 'figures')
plot_profile_likelihood(modelType='hh', filepath=filepath, debug=debug)

plot_profile_likelihood(modelType='mm', filepath=filepath, debug=debug)
plot_profile_likelihood(modelType='mm', sharey=False, filepath=filepath, debug=debug)

plot_profile_likelihood(modelType='ikr', filepath=filepath, debug=debug)

plot_profile_likelihood(modelType='ikur', filepath=filepath, debug=debug)

plot_profile_likelihood(modelType='ina', filepath=filepath, debug=debug)

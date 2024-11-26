import ionbench
from ionbench.uncertainty.profile_likelihood import plot_profile_likelihood
import os

debug = True
filepath = os.path.join(ionbench.ROOT_DIR, '..', 'scripts', 'figures')
plot_profile_likelihood(modelType='hh', filepath=filepath, debug=debug)

plot_profile_likelihood(modelType='mm', filepath=filepath, debug=debug)
plot_profile_likelihood(modelType='mm', sharey=False, filepath=filepath, debug=debug)

plot_profile_likelihood(modelType='ikr', filepath=filepath, debug=debug)

plot_profile_likelihood(modelType='ikur', filepath=filepath, debug=debug)

plot_profile_likelihood(modelType='ina', filepath=filepath, debug=debug)

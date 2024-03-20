# %%
# import sys
# sys.path.append("/var/data/mrufolo/Inverting_luminosity/inv_gauss_tree_maker/tree_maker")
import tree_maker
from tree_maker import NodeJob
from tree_maker import initialize
import time
import os
from pathlib import Path
import itertools
import numpy as np
import yaml
from user_defined_functions import generate_run_sh
from user_defined_functions import generate_run_sh_htc


# Read the active environment name from the local environment
active_environment = os.getenv('CONDA_DEFAULT_ENV')

# Load the YAML file
with open('config.yaml', 'r') as file:
    data = yaml.safe_load(file)

# Update the 'setup_env_script' value with the active environment name
data['root']['setup_env_script'] = f'activate {active_environment}'

# Serialize the updated data structure back into a YAML file
with open('local_config.yaml', 'w') as file:
    yaml.dump(data, file)


# Import the configuration
config=yaml.safe_load(open('local_config.yaml'))

# The user defines the variable to scan
# machine parameters scans




children= {}

number_of_jobs = 1

n_iterations = 100
for jobs in range(number_of_jobs):
    children[f'{jobs:03}_child'] = {
                            'number_of_iterations':n_iterations,
                            'core':0,
                        }

    

if config['root']['use_yaml_children']== False:
    config['root']['children'] = children
#config['root']['setup_env_script'] = os.getcwd() + '/../miniconda/bin/activate'

# Create tree object
start_time = time.time()
root = initialize(config)
print('Done with the tree creation.')
print("--- %s seconds ---" % (time.time() - start_time))

# From python objects we move the nodes to the file-system.
start_time = time.time()
#root.make_folders(generate_run_sh)
root.make_folders(generate_run_sh_htc)
print('The tree folders are ready.')
print("--- %s seconds ---" % (time.time() - start_time))

# %%

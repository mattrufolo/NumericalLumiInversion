#%%
import numpy as np
import random
from matplotlib import pyplot as plt
import json
from numpy import linalg as LA
from scipy.optimize import fsolve, minimize, minimize_scalar, least_squares, root
from inversion_fun import Inv_gauss_xy12 as  inv_g
import yaml
import tree_maker

######
# Open the config.yaml in order to read the config file
with open('config.yaml', 'r') as file:
    cfg = yaml.safe_load(file)

# Start tree_maker logging if log_file is present in config
# # try:
if 'log_file' not in cfg.keys():
    tree_maker = None
# except:
#     tree_maker = None

if tree_maker is not None:
    tree_maker.tag_json.tag_it(cfg['log_file'], 'started')


# set the seed for reproducibility 
random.seed(cfg['seed'])
dict_shift = {'mu0x':[0.01],'mu0y':[0.01],'px':[0.01],'py':[0.01]}#,'betax':[0.01],'betay':[0.01],'alphax':[0.01],'alphay':[0.01],'sigmaz':[0.01]}#cfg['dict_shift']

LS = []
eps = []

## solution can be 2 or 4 depending on the number of unkowns
sol_LS = np.zeros([cfg['number_of_iterations'],4])

## the values of the different component of the LS, depending on how many
## equations are in the system
# f_sol = np.zeros([cfg['number_of_iterations'],len(dict_shift)+1])
# f_sol = np.zeros([cfg['number_of_iterations'],2])
f_sol = np.zeros([cfg['number_of_iterations'],9604])#two parameters

## this is the Jacobian associated to that system
# Jac_sol = np.zeros([cfg['number_of_iterations'],len(dict_shift)+1,2])
# Jac_sol = np.zeros([cfg['number_of_iterations'],2,2])
Jac_sol = np.zeros([cfg['number_of_iterations'],9604,4])#two parameters

delta_sigmaz = {}


## a cycle on the number of iterations for the different random choices of 
## emittance.
for i in range(cfg['number_of_iterations']):
    epsx1 = random.uniform(2.3e-6-4.6e-7,2.3e-6+4.6e-7)
    epsy1 = random.uniform(2.3e-6-4.6e-7,2.3e-6+4.6e-7)
    epsx2 = random.uniform(2.3e-6-4.6e-7,2.3e-6+4.6e-7)
    epsy2 = random.uniform(2.3e-6-4.6e-7,2.3e-6+4.6e-7)
    eps.append([epsx1,epsy1,epsx2,epsy2])
    ## algorithm of inversion
    if 'sigmaz' in dict_shift:
        [sol_LS[i],f_sol[i],Jac_sol[i],timing_LS,nfev_LS,delta_sigmaz[f'par_it{i}']] = inv_g.inv_gauss_single_par_randerr(epsx1,epsy1,epsx2,epsy2,dict_shift)
    else:
        [sol_LS[i],f_sol[i],Jac_sol[i],timing_LS,nfev_LS] = inv_g.inv_gauss_single_par_randerr(epsx1,epsy1,epsx2,epsy2,dict_shift)
    LS.append([timing_LS,nfev_LS])

sol_LS = [list(list(sol_LS)[ii]) for ii in range(len(sol_LS))]
f_sol = [list(list(f_sol)[ii]) for ii in range(len(f_sol))]
Jac_sol = [[list(list(Jac_sol)[ii][jj]) for jj in range(len(Jac_sol[ii]))] for ii in range(len(Jac_sol))]
if 'sigmaz' in dict_shift:
    rel_err = {'est_eps': sol_LS, 'eps': eps, 'f_sol': f_sol, 'Jacobian': Jac_sol, 'LS': LS, 'parameters_sigmaz': delta_sigmaz}
else:
    rel_err = {'est_eps': sol_LS, 'eps': eps, 'f_sol': f_sol, 'Jacobian': Jac_sol, 'LS': LS}

## save a json file with the interesting informations.
with open('output_data.json','w') as outfile:
    json.dump(rel_err,outfile)



if tree_maker is not None:
    tree_maker.tag_json.tag_it(cfg['log_file'], 'completed')

# %%

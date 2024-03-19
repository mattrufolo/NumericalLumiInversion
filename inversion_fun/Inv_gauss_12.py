# CHECK IF THE CHANGES IN THE PARAMETERS WOULD LEAD TO AN OFFSET IN THE VALUES OF THE FUNCTION OR THE STRUCTURE IS MORE COMPLEX, IN CASE WHAT IS THIS BEHAVIOUR?
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#NOISE CROSSING ANGLE!!!!!!!!!!!!!! MOVING ONLY ORBIT
#%%
import numpy as np
import random
import matplotlib.colors as mcolors
from matplotlib import pyplot as plt
from numpy import random as rnd

import copy
from numpy import linalg as LA
import time
from scipy.optimize import fsolve, minimize, minimize_scalar, least_squares, root
from inversion_fun import lumi_formula_no_inner as lumi 

print("numbu")
#parameters that we imagine and that we set!
[f,nb,N,energy_tot] = [11245,2736,1.4e11,6800]
[dif_mu0x,dif_mu0y] = [0,0]
[dif_px,dif_py] = [320e-6,0]#[160e-6,160e-6]
[alphax,alphay] = [0,0]
[sigmaz] = [0.35]
[betax,betay] =[0.3,0.3]
[deltap_p0] = [0]
[dmu0x,dmu0y] = [0,0]
[dpx,dpy] = [0,0]


parameters_sym_1 = [f,nb,N,N,energy_tot,energy_tot,dif_mu0x,dif_mu0y,dif_py,dif_px, alphax,alphay,alphax,alphay,sigmaz,sigmaz,
            betax,betay,betax,betay,deltap_p0,deltap_p0,dmu0x,dmu0y,-dmu0x,-dmu0y,dpx,dpy,-dpx,-dpy]

parameters_sym_2 = [f,nb,N,N,energy_tot,energy_tot,dif_mu0x,dif_mu0y,dif_px,dif_py, alphax,alphay,alphax,alphay,sigmaz,sigmaz,
            betax,betay,betax*0.8,betay*0.8,deltap_p0,deltap_p0,dmu0x,dmu0y,-dmu0x,-dmu0y,dpx,dpy,-dpx,-dpy]

# positions of the parameters!!!
dict_par = {'f':0,'nb':1,'N1':2,'N2':3,'energy_tot1':4,'energy_tot2':5,'mu0x':6,'mu0y':7,'px':8,'py':9,
                    'alphax':10,'alphay':11,'sigmaz':14,
                    'betax':16,'betay':17,'deltap_p0_1':20,'deltap_p0_2':21,
                    'dmu0x_1':22,'dmu0y_1':23,'dmu0x_2':24,'dmu0y_2':25,'dpx_1':26,'dpy_1':27,'dpx_2':28,'dpy_2':29}


#Luminosity that we imagine and the real one!
def L_over_parameters_sym(epsx1,epsy1,epsx2,epsy2, par = parameters_sym_1):

    return lumi.L(epsx1,epsy1,epsx2,epsy2,*par)

# computation in order to obtain the shift in order to change the original luminosity of a percentage!
def percent_sym_short(epsx1,epsy1,epsx2,epsy2,delta,param,m, par = parameters_sym_1):
    delta_param = copy.copy(par)
    delta_param[dict_par[param]] += delta
    len(par)

    if param in ['betax','betay','alphax','alphay']:
        delta_param[dict_par[param]+2] += delta
    if param in ['sigmaz']:
        delta_param[dict_par[param]+1] += delta
    c = (L_over_parameters_sym(epsx1,epsy1,epsx2,epsy2, par = delta_param)-L_over_parameters_sym(epsx1,epsy1,epsx2,epsy2,par=par))
    if c!=0:
        sign = np.sign(c)
    else: 
        sign = +1
    return abs(c/L_over_parameters_sym(epsx1,epsy1,epsx2,epsy2)+sign*(-m))

def inv_gauss_12(eps1,eps2,dict_shift,nfev = 3000, verbose = 0, par = parameters_sym_1):
    '''
    Inversion the luminosity function in order to obtain the emittances, in the case in which the model
    is coherent and also the parameters, shifting the parameters in the dict_shift
    but once a time, can be generalized to multiple shift at the same time wrt to the reference configuration.

    dict_shift: is a dictionary in which the keys are the parameters that we want to change, 
    and the values is the change in percentage that we want to see in the original luminosity

    '''
 
    delta_par = {}
    for i in range(len(dict_shift)):
        param = list(dict_shift.keys())[i]
        print(param)
        print(dict_shift[param])
        for j in range(len(dict_shift[param])):
            shift_par= fsolve(lambda x:percent_sym_short(eps1,eps1,eps2,eps2,x,param,dict_shift[param][j]), x0 = 0)[0]
            delta_par[f'{dict_shift[param][j]}{param}'] = copy.copy(par)
            delta_par[f'{dict_shift[param][j]}{param}'][dict_par[param]]+= shift_par
            if param in ['betax','betay','alphax','alphay']:
                delta_par[f'{dict_shift[param][j]}{param}'][dict_par[param]+2]+= shift_par
            if param in ['sigmaz']:
                delta_par[f'{dict_shift[param][j]}{param}'][dict_par[param]+1]+= shift_par
    print({'par' :delta_par})
    constants = [L_over_parameters_sym(eps1,eps1,eps2,eps2)]
    for i in range(len(dict_shift)):
        param = list(dict_shift.keys())[i]
        for j in range(len(dict_shift[param])):
            constants = np.concatenate([constants,[L_over_parameters_sym(eps1,eps1,eps2,eps2, par = delta_par[f'{dict_shift[param][j]}{param}'])]])
    print(f"{ ', '.join(map(str, constants))}")
    


    def func_to_zero(x):
        output = [L_over_parameters_sym(x[0],x[0],x[1],x[1])-constants[0]]
        for i in range(len(dict_shift)):
            param = list(dict_shift.keys())[i]
            for j in range(len(dict_shift[param])):
                output = np.concatenate([output,[L_over_parameters_sym(x[0],x[0],x[1],x[1], par = delta_par[f'{dict_shift[param][j]}{param}'])-\
                    constants[i*(len(dict_shift[param]))+1+j]]])
        return output
    if verbose == 2:
        def log_iteration(x):
            print('=============')
            print(f"Solution: [{ ', '.join(map(str, x))}]")
            print("Objective:", func_to_zero(x))
            print()
        def func_to_zero_log(x):
            log_iteration(x)
            return func_to_zero(x)
        start_LS = time.time()
        roots = least_squares(func_to_zero_log,x0 = [2.3e-6,2.3e-6],bounds = ([1.5e-6,1.5e-6],[3e-6,3e-6]),max_nfev = nfev, xtol = 1e-16)
        end_LS = time.time()
        time_LS = end_LS-start_LS
    else: 
        start_LS = time.time()
        roots = least_squares(func_to_zero,x0 = [2.3e-6,2.3e-6],bounds = ([1.5e-6,1.5e-6],[3e-6,3e-6]),max_nfev = nfev, xtol = 1e-16)
        end_LS = time.time()
        time_LS = end_LS-start_LS

    par_inter = 'sigmaz'

    if par_inter in dict_shift:
        return [roots.x,roots.fun, roots.jac, time_LS, roots.nfev,[delta_par[f'{dict_shift[par_inter][j]}{par_inter}'][18:20] for j in range(len(dict_shift[par_inter]))]]
    else:
        return [roots.x,roots.fun, roots.jac, time_LS, roots.nfev]

# #######################################

def inv_gauss_12_randerr(eps1,eps2,dict_shift,nfev = 3000, verbose = 0, iteration  = 0):
    '''
    Inversion the luminosity function in order to obtain the emittances, in the case in which the model
    is coherent and also the parameters, shifting the parameters in the dict_shift
    but once a time, can be generalized to multiple shift at the same time wrt to the reference configuration.

    dict_shift: is a dictionary in which the keys are the parameters that we want to change, 
    and the values is the change in percentage that we want to see in the original luminosity

    '''
    print("u")
    iteration*=0
    n_shifts = 10
    delta_parx = {}
    delta_pary = {}
    index_par = 0
    for parx in [parameters_sym_1,parameters_sym_2]:
        delta_par = {}
        index_par +=1
        for i in range(len(dict_shift)):
            param = list(dict_shift.keys())[i]
            print(param)
            print(dict_shift[param])
            for j in range(len(dict_shift[param])):
                goal_shift_par= fsolve(lambda x:percent_sym_short(eps1,eps1,eps2,eps2,x,param,dict_shift[param][j],par = parx), x0 = 0)[0]
                # three_svd_par = abs(av_shift_par)/100
                # shift_par = rnd.normal(av_shift_par,three_svd_par/3,1)[0]
                # # a loop in order to not obtain a negative value on betax or sigmaz 
                # if param in ['betax','betay','sigmaz'] and shift_par + parameters_sym[dict_par[param]]<0:
                #     shift_par  = abs(shift_par)
                # print(shift_par)
                vec_shift = np.linspace(-goal_shift_par,goal_shift_par,n_shifts)
                for k in range(n_shifts):
                    shift_par = vec_shift[k]
                    delta_par[f'{dict_shift[param][j]}{param}_{k}'] = copy.copy(parx)
                    delta_par[f'{dict_shift[param][j]}{param}_{k}'][dict_par[param]]+= shift_par
                    if param in ['betax','betay','alphax','alphay']:
                        delta_par[f'{dict_shift[param][j]}{param}_{k}'][dict_par[param]+2]+= shift_par
                    if param in ['sigmaz']:
                        delta_par[f'{dict_shift[param][j]}{param}_{k}'][dict_par[param]+1]+= shift_par
        if index_par ==1:
            delta_par1 = delta_par
        else:
            delta_par2 = delta_par
    print({'par1' :delta_par1})
    print({'par2' :delta_par2})
    rnd.seed(10)
    L_par = L_over_parameters_sym(eps1,eps1,eps2,eps2, par= parameters_sym_1)
    av_random_noise = 0
    svd_random_noise = L_par/1e3
    random_noise = rnd.normal(av_random_noise,svd_random_noise,1)[0]
    while random_noise + L_par <0:
        random_noise = rnd.normal(av_random_noise,svd_random_noise,1)[0]
    print(L_par)
    print(random_noise + L_par)
    constants = [ L_par+random_noise]

    L_par = L_over_parameters_sym(eps1,eps1,eps2,eps2, par= parameters_sym_2)
    svd_random_noise = L_par/1e3
    random_noise = rnd.normal(av_random_noise,svd_random_noise,1)[0]
    while random_noise + L_par <0:
        random_noise = rnd.normal(av_random_noise,svd_random_noise,1)[0]
    constants = np.concatenate([constants,[L_par+random_noise]])
    
        
    for i in range(len(dict_shift)):
        param = list(dict_shift.keys())[i]
        for j in range(len(dict_shift[param])):
            for k in range(n_shifts):
                for delta_par in [delta_par1,delta_par2]:    
                    L_par = L_over_parameters_sym(eps1,eps1,eps2,eps2, par = delta_par[f'{dict_shift[param][j]}{param}_{k}'])
                    svd_random_noise = L_par/1e3
                    random_noise = rnd.normal(av_random_noise,svd_random_noise,1)[0]
                    print(L_par)
                    print(random_noise + L_par)
                    while random_noise + L_par <0:
                        random_noise = rnd.normal(av_random_noise,svd_random_noise,1)[0]
                    constants = np.concatenate([constants,[L_par+random_noise]])
    print(f"{ ', '.join(map(str, constants))}")

    def func_to_zero(x):
        output = [L_over_parameters_sym(x[0],x[0],x[1],x[1],par= parameters_sym_1)-constants[0]]
        output = np.concatenate([output,[L_over_parameters_sym(x[0],x[0],x[1],x[1],par= parameters_sym_2)-constants[1]]])
        for i in range(len(dict_shift)):
            param = list(dict_shift.keys())[i]
            for j in range(len(dict_shift[param])):
                for k in range(n_shifts):
                    for index_par in range(2):
                        print(len(constants))
                        print(2*(i*n_shifts+1+k)+index_par)
                        print(constants[2*(i*n_shifts+1+k)+index_par])
                        delta_par = [delta_par1,delta_par2][index_par]
                        output = np.concatenate([output,[L_over_parameters_sym(x[0],x[0],x[1],x[1], par = delta_par[f'{dict_shift[param][j]}{param}_{k}'])-\
                            constants[2*(i*n_shifts+1+k)+index_par]]])
        return output
    
    if verbose == 2:
        def log_iteration(x):
            print('=============')
            print(f"Solution: [{ ', '.join(map(str, x))}]")
            print("Objective:", func_to_zero(x))

            return iteration
        def func_to_zero_log(x):
            iter = log_iteration(x)
            return func_to_zero(x)
        start_LS = time.time()
        
        roots = least_squares(func_to_zero_log,x0 = [2.3e-6,2.3e-6],bounds = ([0.8*2.3e-6,0.8*2.3e-6],[1.2*2.3e-6,1.2*2.3e-6]),max_nfev = nfev, xtol = 1e-16)
        end_LS = time.time()
        time_LS = end_LS-start_LS
        print('=============')
        print(f"Solution: [{ ', '.join(map(str, roots.x))}]")
        print("Objective:", roots.fun)

    else: 
        start_LS = time.time()
        
        roots = least_squares(func_to_zero,x0 = [2.3e-6,2.3e-6],bounds = ([0.8*2.3e-6,0.8*2.3e-6],[1.2*2.3e-6,1.2*2.3e-6]),max_nfev = nfev, xtol = 1e-16)
        end_LS = time.time()
        time_LS = end_LS-start_LS
        print('=============')
        print(f"Solution: [{ ', '.join(map(str, roots.x))}]")
        print("Objective:", roots.fun)

    #print(time_LS)
    print(roots.nfev)
    print(roots.message)
    print(roots.fun)

    return [roots.x,roots.fun, roots.jac, time_LS, roots.nfev]

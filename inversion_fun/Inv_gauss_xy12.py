# CHECK IF THE CHANGES IN THE PARAMETERS WOULD LEAD TO AN OFFSET IN THE VALUES OF THE FUNCTION OR THE STRUCTURE IS MORE COMPLEX, IN CASE WHAT IS THIS BEHAVIOUR?
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#NOISE CROSSING ANGLE!!!!!!!!!!!!!! MOVING ONLY ORBIT
#%%
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

# parameters_sym = [f,nb,N,N,energy_tot,energy_tot,dif_mu0x,dif_mu0y,dif_px,dif_py, alphax,alphay,alphax,alphay,sigmaz,sigmaz,
#             betax,betay,betax,betay,deltap_p0,deltap_p0,dmu0x,dmu0y,-dmu0x,-dmu0y,dpx,dpy,-dpx,-dpy]


parameters_sym_x = [f,nb,N,N,energy_tot,energy_tot,dif_mu0x,dif_mu0y,dif_px,dif_py, alphax,alphay,alphax,alphay,sigmaz,sigmaz,
            betax,betay,betax+1e-4,betay+1e-4,deltap_p0,deltap_p0,dmu0x,dmu0y,-dmu0x,-dmu0y,dpx,dpy,-dpx,-dpy]

parameters_sym_y = [f,nb,N,N,energy_tot,energy_tot,dif_mu0x,dif_mu0y,dif_py,dif_px, alphax,alphay,alphax,alphay,sigmaz,sigmaz,
            betax,betay,betax,betay,deltap_p0,deltap_p0,dmu0x,dmu0y,-dmu0x,-dmu0y,dpx,dpy,-dpx,-dpy]


parameters_sym_x2 = [f,nb,N,N,energy_tot,energy_tot,dif_mu0x,dif_mu0y,dif_px,dif_py, alphax,alphay,alphax,alphay,sigmaz,sigmaz,
             betax,betay,betax*0.98,betay*0.98,deltap_p0,deltap_p0,dmu0x,dmu0y,-dmu0x,-dmu0y,dpx,dpy,-dpx,-dpy]

parameters_sym_y2 = [f,nb,N,N,energy_tot,energy_tot,dif_mu0x,dif_mu0y,dif_py,dif_px, alphax,alphay,alphax,alphay,sigmaz,sigmaz,
             betax,betay,betax*0.98,betay*0.98,deltap_p0,deltap_p0,dmu0x,dmu0y,-dmu0x,-dmu0y,dpx,dpy,-dpx,-dpy]

#parameters_sym_x = parameters_sym_y2
# positions of the parameters!!!
dict_par = {'f':0,'nb':1,'N1':2,'N2':3,'energy_tot1':4,'energy_tot2':5,'mu0x':6,'mu0y':7,'px':8,'py':9,
                    'alphax':10,'alphay':11,'sigmaz':14,
                    'betax':16,'betay':17,'deltap_p0_1':20,'deltap_p0_2':21,
                    'dmu0x_1':22,'dmu0y_1':23,'dmu0x_2':24,'dmu0y_2':25,'dpx_1':26,'dpy_1':27,'dpx_2':28,'dpy_2':29}




#Luminosity that we imagine and the real one!
def L_over_parameters_sym(epsx1,epsy1,epsx2,epsy2, par = parameters_sym_x):

    return lumi.L(epsx1,epsy1,epsx2,epsy2,*par)

def L_over_parameters_sym_noise(epsx1,epsy1,epsx2,epsy2,noise_px1, par = parameters_sym_x):
        par_err = copy.copy(par)
        par_err[dict_par['px']] += noise_px1
        # par_err[dict_par['px']+2] += noise_px2
        # par_err[dict_par['py']] += noise_py1
        # par_err[dict_par['py']+2] += noise_py2
        #print(par)
        return lumi.L(epsx1,epsy1,epsx2,epsy2,*par_err)

# def L_over_parameters_sym_noise(epsx1,epsy1,epsx2,epsy2,noise_px1, par = parameters_sym):
#     par_err = copy.copy(par)
#     par_err[dict_par['px']] += noise_px1
#     # par_err[dict_par['px']+2] += noise_px2
#     # par_err[dict_par['py']] += noise_py1
#     # par_err[dict_par['py']+2] += noise_py2
#     #print(par)
#     return lumi.L(epsx1,epsy1,epsx2,epsy2,*par_err)

# computation in order to obtain the shift in order to change the original luminosity of a percentage!
def percent_sym_short(epsx1,epsy1,epsx2,epsy2,delta,param,m):
    delta_param = copy.copy(parameters_sym_x)
    delta_param[dict_par[param]] += delta
    # if param in ['mu0x','mu0y','px','py']:
    #     delta_param[dict_par[param]+2] -= delta
    if param in ['betax','betay','alphax','alphay']:
        delta_param[dict_par[param]+2] += delta
    if param in ['sigmaz']:
        delta_param[dict_par[param]+1] += delta
    c = (L_over_parameters_sym(epsx1,epsy1,epsx2,epsy2, par = delta_param)-L_over_parameters_sym(epsx1,epsy1,epsx2,epsy2))
    if c!=0:
        sign = np.sign(c)
    else: 
        sign = +1
    return abs(c/L_over_parameters_sym(epsx1,epsy1,epsx2,epsy2)+sign*(-m))

iteration = 0
def inv_gauss_single_par(epsx1,epsy1,epsx2,epsy2,dict_shift,nfev = 3000, verbose = 0, iteration  = 0):
    '''
    Inversion the luminosity function in order to obtain the emittances, in the case in which the model
    is coherent and also the parameters, shifting the parameters in the dict_shift
    but once a time, can be generalized to multiple shift at the same time wrt to the reference configuration.

    dict_shift: is a dictionary in which the keys are the parameters that we want to change, 
    and the values is the change in percentage that we want to see in the original luminosity

    '''
    
    iteration*=0

    delta_par = {'':copy.copy(parameters_sym_x)}
    for i in range(len(dict_shift)):
        param = list(dict_shift.keys())[i]
        print(param)
        print(dict_shift[param])
        for j in range(len(dict_shift[param])):
            shift_par= fsolve(lambda x:percent_sym_short(epsx1,epsy1,epsx2,epsy2,x,param,dict_shift[param][j]), x0 = 0)[0]
            # three_svd_par = abs(av_shift_par)/100
            # shift_par = rnd.normal(av_shift_par,three_svd_par/3,1)[0]
            # # a loop in order to not obtain a negative value on betax or sigmaz 
            # if param in ['betax','betay','sigmaz'] and shift_par + parameters_sym[dict_par[param]]<0:
            #     shift_par  = abs(shift_par)
            # print(shift_par)
            delta_par[f'{dict_shift[param][j]}{param}'] = copy.copy(parameters_sym_x)
            delta_par[f'{dict_shift[param][j]}{param}'][dict_par[param]]+= shift_par
            # # if param in ['mu0x','mu0y','px','py']:
            # #     delta_par[f'{dict_shift[param][j]}{param}'][dict_par[param]+2]-= shift_par
            if param in ['betax','betay','alphax','alphay']:
                delta_par[f'{dict_shift[param][j]}{param}'][dict_par[param]+2]+= shift_par
            if param in ['sigmaz']:
                delta_par[f'{dict_shift[param][j]}{param}'][dict_par[param]+1]+= shift_par
            # this is for do combination of the shifts all togheter, to get more combination
            # for k in range(len(delta_par)):
            #     keys = list(delta_par.keys())
            #     delta_par[keys[k]+f'{dict_shift[param][j]}{param}'] = copy.copy(delta_par[keys[k]])
            #     delta_par[keys[k]+f'{dict_shift[param][j]}{param}'][dict_par[param]]+= shift_par
            #     # if param in ['mu0x','mu0y','px','py']:
            #     #     delta_par[f'{dict_shift[param][j]}{param}'][dict_par[param]+2]-= shift_par
            #     if param in ['betax','betay','alphax','alphay']:
            #         delta_par[keys[k]+f'{dict_shift[param][j]}{param}'][dict_par[param]+2]+= shift_par
            #     if param in ['sigmaz']:
            #         delta_par[keys[k]+f'{dict_shift[param][j]}{param}'][dict_par[param]+1]+= shift_par
    print({'par' :delta_par})
    constants = [L_over_parameters_sym(epsx1,epsy1,epsx2,epsy2)]
    for i in range(len(delta_par)-1):
        param = list(delta_par.keys())[i+1]
        L_par = L_over_parameters_sym(epsx1,epsy1,epsx2,epsy2, par = delta_par[f'{param}'])
        constants = np.concatenate([constants,[L_par]])
    print(f"{ ', '.join(map(str, constants))}")
    


    def func_to_zero(x):
        output = [L_over_parameters_sym(x[0],x[1],x[2],x[3])-constants[0]]
        for i in range(len(delta_par)-1):
            param = list(delta_par.keys())[i+1]
            output = np.concatenate([output,[L_over_parameters_sym(x[0],x[1],x[2],x[3], par = delta_par[f'{param}'])-\
                constants[i+1]]])
        return output
    if verbose == 2:
        def log_iteration(x):
            print('=============')
            print(f"Solution: [{ ', '.join(map(str, x))}]")
            print("Objective:", func_to_zero(x))
            print()

            return iteration
        def func_to_zero_log(x):
            iter = log_iteration(x)
            return func_to_zero(x)
        start_LS = time.time()
        roots = least_squares(func_to_zero_log,x0 = [2.3e-6,2.3e-6,2.3e-6,2.3e-6],bounds = ([1.5e-6,1.5e-6,1.5e-6,1.5e-6],[3e-6,3e-6,3e-6,3e-6]),max_nfev = nfev, xtol = 1e-16)
        end_LS = time.time()
        time_LS = end_LS-start_LS
    else: 
        start_LS = time.time()
        roots = least_squares(func_to_zero,x0 = [2.3e-6,2.3e-6,2.3e-6,2.3e-6],bounds = ([1.5e-6,1.5e-6,1.5e-6,1.5e-6],[3e-6,3e-6,3e-6,3e-6]),max_nfev = nfev, xtol = 1e-16)
        end_LS = time.time()
        time_LS = end_LS-start_LS
    #print(time_LS)
    #print(roots.nfev)
    #print(roots.message)
    #print(roots.fun)
    #start = time.time()
    #func_to_zero([epsx1,epsy1,epsx2,epsy2])
    #end = time.time()
    #print(end-start)
    #print( LA.norm((np.array(roots.x[:4])-np.array([epsx1,epsy1,epsx2,epsy2]))/LA.norm(np.array([epsx1,epsy1,epsx2,epsy2]))))
    
    par_inter = 'sigmaz'

    if par_inter in dict_shift:
#        print([delta_par[f'{dict_shift[par_inter][j]}{par_inter}'][18:20] for j in range(len(dict_shift[par_inter]))])
        return [roots.x,roots.fun, roots.jac, time_LS, roots.nfev,[delta_par[f'{dict_shift[par_inter][j]}{par_inter}'][18:20] for j in range(len(dict_shift[par_inter]))]]
    else:
        return [roots.x,roots.fun, roots.jac, time_LS, roots.nfev]

########################################
def inv_gauss_single_par_randerr(epsx1,epsy1,epsx2,epsy2,dict_shift,nfev = 3000, verbose = 0, iteration  = 0):
    '''
    Inversion the luminosity function in order to obtain the emittances, in the case in which the model
    is coherent and also the parameters, shifting the parameters in the dict_shift
    but once a time, can be generalized to multiple shift at the same time wrt to the reference configuration.

    dict_shift: is a dictionary in which the keys are the parameters that we want to change, 
    and the values is the change in percentage that we want to see in the original luminosity

    '''
    print("ua")
    iteration*=0
    n_shifts = 9
    delta_parx = {'': parameters_sym_x}
    delta_pary = {'': parameters_sym_y}
    delta_parx2 = {'': parameters_sym_x2}
    delta_pary2 = {'': parameters_sym_y2}
    ## delta_par_sym = {}
    index_par = 0
    print(parameters_sym_x)
    #print(parameters_sym_y)
    # for par in [parameters_sym_x,parameters_sym_y,parameters_sym]:
    for i in range(len(dict_shift)):
        param = list(dict_shift.keys())[i]
        print(param)
        print(dict_shift[param])
        for j in range(len(dict_shift[param])):
            goal_shift_par= fsolve(lambda x:percent_sym_short(epsx1,epsy1,epsx2,epsy2,x,param,dict_shift[param][j]), x0 = 0)[0]
            # three_svd_par = abs(av_shift_par)/100
            # shift_par = rnd.normal(av_shift_par,three_svd_par/3,1)[0]
            # # a loop in order to not obtain a negative value on betax or sigmaz 
            # if param in ['betax','betay','sigmaz'] and shift_par + parameters_sym[dict_par[param]]<0:
            #     shift_par  = abs(shift_par)
            # print(shift_par)
            vec_shift = np.linspace(-goal_shift_par,goal_shift_par,n_shifts)
            for p in range(len(delta_parx)):
                keys = list(delta_parx.keys())
                for k in range(n_shifts):
                    shift_par = vec_shift[k]
                    delta_parx[keys[p]+f'{dict_shift[param][j]}{param}_{k}'] = copy.copy(delta_parx[keys[p]])
                    delta_parx[keys[p]+f'{dict_shift[param][j]}{param}_{k}'][dict_par[param]]+= shift_par
                    delta_pary[keys[p]+f'{dict_shift[param][j]}{param}_{k}'] = copy.copy(delta_pary[keys[p]])
                    delta_pary[keys[p]+f'{dict_shift[param][j]}{param}_{k}'][dict_par[param]]+= shift_par
                    delta_parx2[keys[p]+f'{dict_shift[param][j]}{param}_{k}'] = copy.copy(delta_parx2[keys[p]])
                    delta_parx2[keys[p]+f'{dict_shift[param][j]}{param}_{k}'][dict_par[param]]+= shift_par
                    delta_pary2[keys[p]+f'{dict_shift[param][j]}{param}_{k}'] = copy.copy(delta_pary2[keys[p]])
                    delta_pary2[keys[p]+f'{dict_shift[param][j]}{param}_{k}'][dict_par[param]]+= shift_par
                    ## delta_parx[f'{dict_shift[param][j]}{param}_{k}'] = copy.copy(parameters_sym_x)
                    ## delta_parx[f'{dict_shift[param][j]}{param}_{k}'][dict_par[param]]+= shift_par
                    ## delta_pary[f'{dict_shift[param][j]}{param}_{k}'] = copy.copy(parameters_sym_y)
                    ## delta_pary[f'{dict_shift[param][j]}{param}_{k}'][dict_par[param]]+= shift_par
                    ## delta_parx2[f'{dict_shift[param][j]}{param}_{k}'] = copy.copy(parameters_sym_x2)
                    ## delta_parx2[f'{dict_shift[param][j]}{param}_{k}'][dict_par[param]]+= shift_par
                    ## delta_pary2[f'{dict_shift[param][j]}{param}_{k}'] = copy.copy(parameters_sym_y2)
                    ## delta_pary2[f'{dict_shift[param][j]}{param}_{k}'][dict_par[param]]+= shift_par
                    # if param in ['mu0x','mu0y','px','py']:
                    #     delta_par[f'{dict_shift[param][j]}{param}_{k}'][dict_par[param]+2]-= shift_par
                    if param in ['betax','betay','alphax','alphay']:
                        delta_parx[keys[p]+f'{dict_shift[param][j]}{param}_{k}'][dict_par[param]+2]+= shift_par
                        delta_pary[keys[p]+f'{dict_shift[param][j]}{param}_{k}'][dict_par[param]+2]+= shift_par
                        delta_parx2[keys[p]+f'{dict_shift[param][j]}{param}_{k}'][dict_par[param]+2]+= shift_par
                        delta_pary2[keys[p]+f'{dict_shift[param][j]}{param}_{k}'][dict_par[param]+2]+= shift_par
                        # delta_parx[f'{dict_shift[param][j]}{param}_{k}'][dict_par[param]+2]+= shift_par
                        # delta_pary[f'{dict_shift[param][j]}{param}_{k}'][dict_par[param]+2]+= shift_par
                        # delta_parx2[f'{dict_shift[param][j]}{param}_{k}'][dict_par[param]+2]+= shift_par
                        # delta_pary2[f'{dict_shift[param][j]}{param}_{k}'][dict_par[param]+2]+= shift_par
                    if param in ['sigmaz']:
                        delta_parx[keys[p]+f'{dict_shift[param][j]}{param}_{k}'][dict_par[param]+1]+= shift_par
                        delta_pary[keys[p]+f'{dict_shift[param][j]}{param}_{k}'][dict_par[param]+1]+= shift_par
                        delta_parx2[keys[p]+f'{dict_shift[param][j]}{param}_{k}'][dict_par[param]+1]+= shift_par
                        delta_pary2[keys[p]+f'{dict_shift[param][j]}{param}_{k}'][dict_par[param]+1]+= shift_par
                        # delta_parx[f'{dict_shift[param][j]}{param}_{k}'][dict_par[param]+1]+= shift_par
                        # delta_pary[f'{dict_shift[param][j]}{param}_{k}'][dict_par[param]+1]+= shift_par
                        # delta_parx2[f'{dict_shift[param][j]}{param}_{k}'][dict_par[param]+1]+= shift_par
                        # delta_pary2[f'{dict_shift[param][j]}{param}_{k}'][dict_par[param]+1]+= shift_par

        #     delta_par_sym = delta_par
    print({'parx' :delta_parx})
    print({'pary' :delta_pary})
    print({'parx2' :delta_parx2})
    print({'pary2' :delta_pary2})
    # print({'par' :delta_par_sym})
    #rnd.seed(10)
    L_par = L_over_parameters_sym(epsx1,epsy1,epsx2,epsy2, par= parameters_sym_x)
    av_random_noise = 0
    svd_random_noise = L_par/1e3
    random_noise = rnd.normal(av_random_noise,svd_random_noise,1)[0]
    while random_noise + L_par <0:
        random_noise = rnd.normal(av_random_noise,svd_random_noise,1)[0]
    print(L_par)
    print(random_noise + L_par)
    constants = [random_noise + L_par]

    L_par = L_over_parameters_sym(epsx1,epsy1,epsx2,epsy2, par= parameters_sym_y)
    svd_random_noise = L_par/1e3
    random_noise = rnd.normal(av_random_noise,svd_random_noise,1)[0]
    while random_noise + L_par <0:
        random_noise = rnd.normal(av_random_noise,svd_random_noise,1)[0]
    constants = np.concatenate([constants,[L_par+random_noise]])

    L_par = L_over_parameters_sym(epsx1,epsy1,epsx2,epsy2, par= parameters_sym_x2)
    svd_random_noise = L_par/1e3
    random_noise = rnd.normal(av_random_noise,svd_random_noise,1)[0]
    while random_noise + L_par <0:
        random_noise = rnd.normal(av_random_noise,svd_random_noise,1)[0]
    constants = np.concatenate([constants,[L_par+random_noise]])

    L_par = L_over_parameters_sym(epsx1,epsy1,epsx2,epsy2, par= parameters_sym_y2)
    svd_random_noise = L_par/1e3
    random_noise = rnd.normal(av_random_noise,svd_random_noise,1)[0]
    while random_noise + L_par <0:
        random_noise = rnd.normal(av_random_noise,svd_random_noise,1)[0]
    constants = np.concatenate([constants,[L_par+random_noise]])

    # L_par = L_over_parameters_sym(epsx1,epsy1,epsx2,epsy2, par= parameters_sym)
    # svd_random_noise = L_par/1e3
    # random_noise = rnd.normal(av_random_noise,svd_random_noise,1)[0]
    # while random_noise + L_par <0:
    #     random_noise = rnd.normal(av_random_noise,svd_random_noise,1)[0]
    # constants = np.concatenate([constants,[L_par+random_noise]])
    for i in range(len(delta_parx)-1):
        param = list(delta_parx.keys())[i+1]
        for delta_par in [delta_parx,delta_pary,delta_parx2,delta_pary2]:
            L_par = L_over_parameters_sym(epsx1,epsy1,epsx2,epsy2, par = delta_par[f'{param}'])
            svd_random_noise = L_par/1e3
            random_noise = rnd.normal(av_random_noise,svd_random_noise,1)[0]
            print(L_par)
            print(random_noise + L_par)
            while random_noise + L_par <0:
                random_noise = rnd.normal(av_random_noise,svd_random_noise,1)[0]
            constants = np.concatenate([constants,[L_par+random_noise]])

    print(f"{ ', '.join(map(str, constants))}")
    print(len(constants))

    
    # output = [L_over_parameters_sym(x[0],x[1],x[2],x[3])-constants[0]]
    #     for i in range(len(delta_par)-1):
    #         param = list(delta_par.keys())[i+1]
    #         output = np.concatenate([output,[L_over_parameters_sym(x[0],x[1],x[2],x[3], par = delta_par[f'{param}'])-\
    #             constants[i+1]]])
    def func_to_zero(x):
        output = [L_over_parameters_sym(x[0],x[1],x[2],x[3],par= parameters_sym_x)-constants[0]]
        output = np.concatenate([output,[L_over_parameters_sym(x[0],x[1],x[2],x[3],par= parameters_sym_y)-constants[1]]])
        output = np.concatenate([output,[L_over_parameters_sym(x[0],x[1],x[2],x[3],par= parameters_sym_x2)-constants[2]]])
        output = np.concatenate([output,[L_over_parameters_sym(x[0],x[1],x[2],x[3],par= parameters_sym_y2)-constants[3]]])
        # output = np.concatenate([output,[L_over_parameters_sym(x[0],x[1],x[2],x[3],par= parameters_sym)-constants[2]]])
        for i in range(len(delta_parx)-1):
            param = list(delta_parx.keys())[i+1]
            for index_par in range(4):
                # delta_par = [delta_parx,delta_pary,delta_par_sym][index_par]
                delta_par = [delta_parx,delta_pary,delta_parx2,delta_pary2][index_par]
                output = np.concatenate([output,[L_over_parameters_sym(x[0],x[1],x[2],x[3], par = delta_par[f'{param}'])-\
                    constants[4*(i+1)+index_par]]])    
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
        roots = least_squares(func_to_zero_log,x0 = [2.3e-6,2.3e-6,2.3e-6,2.3e-6],bounds = ([0.8*2.3e-6,0.8*2.3e-6,0.8*2.3e-6,0.8*2.3e-6],[1.2*2.3e-6,1.2*2.3e-6,1.2*2.3e-6,1.2*2.3e-6]),max_nfev = nfev, xtol = 1e-16)
        end_LS = time.time()
        time_LS = end_LS-start_LS
        print('=============')
        print(f"Solution: [{ ', '.join(map(str, roots.x))}]")
        print("Objective:", roots.fun)

    else: 
        start_LS = time.time()
        roots = least_squares(func_to_zero,x0 = [2.3e-6,2.3e-6,2.3e-6,2.3e-6],bounds = ([0.8*2.3e-6,0.8*2.3e-6,0.8*2.3e-6,0.8*2.3e-6],[1.2*2.3e-6,1.2*2.3e-6,1.2*2.3e-6,1.2*2.3e-6]),max_nfev = nfev, xtol = 1e-16)
        end_LS = time.time()
        time_LS = end_LS-start_LS
        print('=============')
        print(f"Solution: [{ ', '.join(map(str, roots.x))}]")
        print("Objective:", roots.fun)

    #print(time_LS)
    #print(roots.nfev)
    #print(roots.message)
    #print(roots.fun)
    #start = time.time()
    #func_to_zero([epsx1,epsy1,epsx2,epsy2])
    #end = time.time()
    #print(end-start)
    #print( LA.norm((np.array(roots.x[:4])-np.array([epsx1,epsy1,epsx2,epsy2]))/LA.norm(np.array([epsx1,epsy1,epsx2,epsy2]))))
    
#     par_inter = 'sigmaz'

#     if par_inter in dict_shift:
# #        print([delta_par[f'{dict_shift[par_inter][j]}{par_inter}'][18:20] for j in range(len(dict_shift[par_inter]))])
#         return [roots.x,roots.fun, roots.jac, time_LS, roots.nfev,[delta_par[f'{dict_shift[par_inter][j]}{par_inter}'][18:20] for j in range(len(dict_shift[par_inter]))]]
#     else:
    return [roots.x,roots.fun, roots.jac, time_LS, roots.nfev]
    
########################################




#%%
# import contextlib
# #dict_shift = {'mu0x':[0.01],'mu0y':[0.01],'px':[0.01],'py':[0.01],'betax':[0.01],'betay':[0.01],'alphax':[0.01],'alphay':[0.01],'sigmaz':[0.01]}#,'py':[0.01,0.02]}
# dict_shift = {'mu0x':[0.01],'mu0y':[0.01],'px':[0.01],'py':[0.01]}
# random.seed(21)
# LS = []
# eps = []
# sol_LS = np.zeros([2+1,4])
# f_sol = np.zeros([2+1,1024])#4*len(dict_shift)*(10)+4])
# Jac_sol = np.zeros([2+1,1024,4])#4*len(dict_shift)*(10)+4,4])
# d = {}
# for i in range(1):
#     epsx1 = random.uniform(0.8*2.3e-6,1.2*2.3e-6)
#     epsy1 = random.uniform(0.8*2.3e-6,1.2*2.3e-6)
#     epsx2 = random.uniform(0.8*2.3e-6,1.2*2.3e-6)
#     epsy2 = random.uniform(0.8*2.3e-6,1.2*2.3e-6)
#     eps.append([epsx1,epsy1,epsx2,epsy2])
#     with open(f"output_zoom.txt","w") as h,contextlib.redirect_stdout(h):
#         [sol_LS[i+1],f_sol[i+1],Jac_sol[i+1],timing_LS,nfev_LS] = inv_gauss_single_par_randerr(epsx1,epsy1,epsx2,epsy2,dict_shift,verbose = 2)
#     LS.append([timing_LS,nfev_LS])
# rel_err = {'est_eps': sol_LS, 'eps': eps, 'f_sol': f_sol, 'Jacobian': Jac_sol, 'LS': LS}



 #%%
# def compute_points_on_line(x1, y1, x2, y2, num_points, y_min=None, y_max=None):
#     # Calculate the slope (m)
#     m = (y2 - y1) / (x2 - x1)
    
#     # Calculate the y-intercept (c)
#     c = y1 - m * x1
    
#     # Determine the range of y-values
#     y_min = min(y1, y2) if y_min is None else min(min(y1, y2), y_min)
#     y_max = max(y1, y2) if y_max is None else max(max(y1, y2), y_max)
    
#     # Compute the corresponding x-values for each y-value within the range
#     points = []
#     for i in range(num_points):
#         y = y_min + (y_max - y_min) * i / (num_points - 1)
#         x = (y - c) / m
#         points.append([x, y])
    
#     return points

# def points_along_line(start_point, direction_vector, num_points):

#     """
#     Generate points along a line given a starting point and a direction vector.
#     :param start_point: The starting point of the line as a NumPy array [x0, y0].
#     :param direction_vector: The direction vector of the line as a NumPy array [dx, dy].
#     :param num_points: The number of points to generate along the line.
#     :return: A list of NumPy arrays representing the points along the line.
#     """
#     t_values = np.linspace(0, 1, num_points)
#     points = [start_point + t * direction_vector for t in t_values]
#     return points


# # %%
# import ast
# import math
# import itertools

# with open('output_zoom.txt') as f:
#     contents = f.read()

# last_key = list(dict_shift.keys())[-1]
# constants = list(eval(contents.split('=============\nSolution: ')[0].split('\n')[-2]))
# paramx = ast.literal_eval(contents.split(f'{last_key}\n[0.01]\n')[1].split('\n')[0])
# paramy = 0
# #paramy = ast.literal_eval(contents.split(f'{last_key}\n[0.01]\n')[1].split('\n')[1])
# #paramx2 = ast.literal_eval(contents.split(f'{last_key}\n[0.01]\n')[1].split('\n')[2])
# #paramy2 = ast.literal_eval(contents.split(f'{last_key}\n[0.01]\n')[1].split('\n')[3])
# #paramsym = ast.literal_eval(contents.split(f'{last_key}\n[0.01]\n')[3].split('\n')[2])


# def L_over_eps_xy(eps1,eps2, par):
#     return L_over_parameters_sym(eps1,eps1,eps2,eps2, par)

# # def func_to_zero(x):
# #     output = [(L_over_parameters_sym(x[0],x[1],x[0],x[1])-constants[0])**2]
# #     # for i in range(len(dict_shift)):
# #     #     keys = list(param['par'].keys())
# #     #     print(keys)
# #     #     for j in range(len(keys)):
# #     parmux = [11245, 2736, 140000000000.0, 140000000000.0, 6800, 6800, 0, -1.9817340820528364e-06, 0.00032, 0, 0, 0, 0, 0, 0.35, 0.35, 0.3, 0.3, 0.3001, 0.3001, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# #     output = np.concatenate([output,[(L_over_parameters_sym(x[0],x[1],x[0],x[1], par = parmux)-\
# #         constants[1])**2]])
# #     return np.sqrt(np.sum(output))


# def func_to_zero(x):
#     #output = [L_over_parameters_sym(x[0],x[1],x[2],x[3],par= parameters_sym_x)-constants[0]]
#     # output = np.concatenate([output,[L_over_parameters_sym(x[0],x[1],x[2],x[3],par= parameters_sym_y)-constants[1]]])
#     # output = np.concatenate([output,[L_over_parameters_sym(x[0],x[1],x[2],x[3],par= parameters_sym_x2)-constants[2]]])
#     # output = np.concatenate([output,[L_over_parameters_sym(x[0],x[1],x[2],x[3],par= parameters_sym_y2)-constants[3]]])
#     output = [L_over_parameters_sym(x[0],x[0],x[1],x[1],par= parameters_sym_y)-constants[0]]
#     keys = list(paramx['par'].keys())
#     ax = ['x','y']#ax = ['x','y','x2','y2']
#     for i in range(int(len(constants))-1):#for i in range(int(len(constants)/4)-1):

#         #for j in range(len(dict_shift[param])):
#         for j in range(1):#for j in range(4):
#             j = 0
#             param = [paramx,paramy][j]#param = [paramx,paramy,paramx2,paramy2][j]
#             output = np.concatenate([output,[L_over_parameters_sym(x[0],x[0],x[1],x[1], par = param[f'par'][keys[i]])-\
#                 constants[(i+1)+j]]])
#     return output 

# #roots1 = least_squares(func_to_zero,x0 = [2.3e-6,2.3e-6],bounds = ([1.5e-6,1.5e-6],[3e-6,3e-6]),max_nfev = 3000, xtol = 1e-22)

# def penalty_square(xy):
#     #penalty = (L_over_parameters_sym(xy[0],xy[1],xy[2],xy[3],parameters_sym_x)-constants[0])**2
#     #penalty += (L_over_parameters_sym(xy[0],xy[1],xy[2],xy[3],parameters_sym_y)-constants[1])**2
#     # penalty += (L_over_parameters_sym(xy[0],xy[1],xy[2],xy[3],parameters_sym_x2)-constants[2])**2
#     # penalty += (L_over_parameters_sym(xy[0],xy[1],xy[2],xy[3],parameters_sym_y2)-constants[3])**2
#     penalty = (L_over_eps_xy(xy[0],xy[1],parameters_sym_y)-constants[0])**2
#     keys = list(paramx['par'].keys())
#     ax = ['x','y']#ax = ['x','y','x2','y2']
#     for i in range(int(len(constants))-1):#for i in range(int(len(constants)/4)-1):
#        for j in range(1):#for j in range(4):
#             j = 0
#             # param = [paramx,paramy,paramsym][j]
#             param = [paramx,paramy][j]#param = [paramx,paramy,paramx2,paramy2][j]
#             #param[f'par{ax[j]}']
#             penalty += (L_over_eps_xy(xy[0],xy[1],param[f'par'][keys[i]])-constants[(i+1)+j])**2
#     return [penalty]



# iter_x = [eval(ii.split('\n')[0])[0] for ii in contents.split('=============\nSolution: ')[1:]]
# iter_y = [eval(ii.split('\n')[0])[1] for ii in contents.split('=============\nSolution: ')[1:]]
# #iter_x2 = [eval(ii.split('\n')[0])[2] for ii in contents.split('=============\nSolution: ')[1:]]
# #iter_y2 = [eval(ii.split('\n')[0])[3] for ii in contents.split('=============\nSolution: ')[1:]]
# L_eps_iter = [penalty_square([iter_x[ii],iter_y[ii]]) for ii in range(len(iter_y))]
# iterations = [iter_x,iter_y]


# vec_epsx = np.linspace(0.8*2.3e-6,1.2*2.3e-6,100)#(iter_x[0],iter_x[1]+(iter_x[1]-iter_x[0]),100)
# vec_epsy = np.linspace(0.8*2.3e-6,1.2*2.3e-6,100)#(iter_y[0],iter_y[2]+(iter_y[2]-iter_y[0]),100)
# #vec_epsx2 = np.linspace(0.8*2.3e-6,1.2*2.3e-6,100)#(iter_x[0],iter_x[1]+(iter_x[1]-iter_x[0]),100)
# #vec_epsy2 = np.linspace(0.8*2.3e-6,1.2*2.3e-6,100)#(iter_y[0],iter_y[2]+(iter_y[2]-iter_y[0]),100)

# vec_eps = [vec_epsx,vec_epsy]#,vec_epsx2,vec_epsy2]

# xv,yv = np.array(np.meshgrid(vec_epsx, vec_epsy))
# Z = [[penalty_square([xv[j][ii],yv[j][ii]]) for ii in range(len(xv[j]))] for j in range(len(xv))]
# L_eps = np.flipud(Z)

# # vec_epsx = np.linspace(0.98*rel_err['eps'][0],1.02*rel_err['eps'][0],100)#(iter_x[0],iter_x[1]+(iter_x[1]-iter_x[0]),100)
# # vec_epsy = np.linspace(0.98*rel_err['eps'][1],1.02*rel_err['eps'][1],100)

# # mesh_xy = np.array(np.meshgrid(vec_epsx, vec_epsy)).T.reshape(-1,2)
# # mesh_x = [ii[0] for ii in mesh_xy]
# # mesh_y = [ii[1] for ii in mesh_xy]
# #L_eps_vec = [penalty_square(ii) for ii in mesh_xy]
# #L_eps_vec = np.divide(L_eps,max(L_eps)*np.ones(len(L_eps)))

# # fig,ax = plt.subplots(2,3, figsize = (20,8))

# # perm_set = [list(ii) for ii in list(itertools.combinations([0,1,2,3], 2))]

# # axis = ['epsilonx1','epsilony1','epsilonx2','epsilony2']

# # for i in range(len(perm_set)):
# #     def penalty_square(x):
# #         bool = [ii in perm_set[i] for ii in [0,1,2,3]]
# #         index = {f'{perm_set[i][j]}': j for j in range(len(perm_set[i])) if bool[perm_set[i][j]]}
# #         print(index)
# #         print([x[index[f'{j}']] if bool[j] else rel_err['eps'][0][j] for j in range(len(bool))])
# #         return penalty([x[index[f'{j}']] if bool[j] else rel_err['eps'][0][j] for j in range(len(bool))])
# #     ax_x = int(i/3)
# #     ax_y = np.mod(i,3)
# #     [vec_epsx,vec_epsy] = [vec_eps[ii] for ii in perm_set[i]]
# #     xv,yv = np.array(np.meshgrid(vec_epsx, vec_epsy))
# #     Z = [[penalty_square([xv[j][ii],yv[j][ii]]) for ii in range(len(xv[j]))] for j in range(len(xv))]
# #     L_eps = np.flipud(Z)
# #     scatter0 = ax[ax_x,ax_y].imshow(L_eps,extent =[vec_epsx[0], vec_epsx[-1], vec_epsy[0], vec_epsy[-1]],norm=colors.LogNorm())#norm=colors.LogNorm( vmin = 10**30, vmax = 10**33))
# #     ax[ax_x,ax_y].scatter(rel_err['eps'][0][perm_set[i][0]],rel_err['eps'][0][perm_set[i][1]],marker = 'x', c = 'black',zorder = 3, alpha = 1)
# #     ax[ax_x,ax_y].set_xlabel(f'{axis[perm_set[i][0]]}')
# #     ax[ax_x,ax_y].set_ylabel(f'{axis[perm_set[i][1]]}')
    
# #     x = 2.3e-6
# #     y = 2.3e-6
# #     h = 1e-6
# #     df_dx = (penalty_square([x + h, y])[0] - penalty_square([x - h, y])[0]) / (2 * h)
# #     df_dy = (penalty_square([x, y + h])[0] - penalty_square([x, y - h])[0]) / (2 * h)

# #     gradient = np.array([df_dx,df_dy])
# #     vector_line = np.array([-gradient[1],gradient[0]])/LA.norm(np.array([-gradient[1],gradient[0]]))
# #     c = points_along_line([rel_err['eps'][0][perm_set[i][0]],rel_err['eps'][0][perm_set[i][1]]],-vector_line*1e-6,100)
# #     d = points_along_line([rel_err['eps'][0][perm_set[i][0]],rel_err['eps'][0][perm_set[i][1]]],vector_line*1e-6,100)

# #     points_line = np.concatenate([c[::-2],d])
# #     x_line = [ii[0] for ii in points_line]
# #     y_line = [ii[1] for ii in points_line]
# #     L_eps_line = [penalty_square([x_line[ii],y_line[ii]]) for ii in range(len(x_line))]
# #     dist = [LA.norm(np.array(points_line[ii])-np.array(points_line[0])) for ii in range(len(x_line))]
# #     #ax[ax_x,ax_y].plot(x_line,y_line,'-', c = 'cyan',zorder = 1, alpha = 0.5,dashes=(5, 10))
# #     ax[ax_x,ax_y].set_xlim(0.8*2.3e-6,1.2*2.3e-6)
# #     ax[ax_x,ax_y].set_ylim(0.8*2.3e-6,1.2*2.3e-6)

# #     cmap2 = plt.cm.get_cmap('plasma')
# #     iter_x = iterations[perm_set[i][0]]
# #     iter_y = iterations[perm_set[i][1]]
# #     scatter1= ax[ax_x,ax_y].scatter(iter_x,iter_y,zorder = 2,c=np.arange(len(iter_x)),cmap = cmap2)

# #     cbar = plt.colorbar(scatter0)
# #     cbar.ax.get_yaxis().labelpad = 15
# #     cbar.set_label( "Penalty values", rotation=270)
# #     cbar = plt.colorbar(scatter1)
# #     cbar.ax.get_yaxis().labelpad = 15
# #     cbar.set_label( "# iteration LS", rotation=270)
    
# #     plt.savefig('proj_eps_penalty_crossx_epsxy12_difbeta.png')



# #%%
# #plt.plot(iter_x[1]*np.ones(100), np.linspace(iter_y[0],iter_y[2]+(iter_y[2]-iter_y[0]),100))
# # points_line = compute_points_on_line(2.3e-6,rel_err['eps'][1],iter_x[len(iter_y)-7],iter_y[len(iter_y)-7],100)#,y_min=2e-6,y_max = 2.6e-6)
# # x_line = [ii[0] for ii in points_line]
# # y_line = [ii[1] for ii in points_line]

# x = 2.3e-6
# y = 2.3e-6
# h = 1e-6
# df_dx = (penalty_square([x + h, y])[0] - penalty_square([x - h, y])[0]) / (2 * h)
# df_dy = (penalty_square([x, y + h])[0] - penalty_square([x, y - h])[0]) / (2 * h)

# gradient = np.array([df_dx,df_dy])
# vector_line = np.array([-gradient[1],gradient[0]])/LA.norm(np.array([-gradient[1],gradient[0]]))
# c = points_along_line([rel_err['eps'][0][0],rel_err['eps'][0][1]],-vector_line*1e-6,100)
# d = points_along_line([rel_err['eps'][0][0],rel_err['eps'][0][1]],vector_line*1e-6,100)
# points_line = np.concatenate([c[::-2],d])

# x_line = [ii[0] for ii in points_line]
# y_line = [ii[1] for ii in points_line]
# L_eps_line = [penalty_square([x_line[ii],y_line[ii]]) for ii in range(len(x_line))]
# dist = [LA.norm(np.array(points_line[ii])-np.array(points_line[0])) for ii in range(len(x_line))]
# # plt.plot(dist,L_eps_line)
# #plt.plot(np.linspace(0.8*2.3e-6,1.2*2.3e-6),rel_err['eps'][1]*np.ones(50), c = 'red')
# #plt.plot(np.linspace(0.8*2.3e-6,1.2*2.3e-6),rel_err['est_eps'][1][1]*np.ones(50), c = 'black')
# # plt.plot(np.linspace(2.3e-6,2.6e-6),(rel_err['eps'][1]-2e-9)*np.ones(50), c = 'black')
# # plt.plot(np.linspace(2.3e-6,2.6e-6),(rel_err['eps'][1]+2e-9)*np.ones(50), c = 'cyan')
# # plt.plot(np.linspace(2.3e-6,2.6e-6),(rel_err['eps'][1]+2e-8)*np.ones(50), c = 'magenta')
# # plt.plot(np.linspace(2.3e-6,2.6e-6),(rel_err['eps'][1]+4e-8)*np.ones(50), c = 'blue')
# mesh_xy = np.array(np.meshgrid(vec_epsx, vec_epsy)).T.reshape(-1,2)
# L_eps_vec = [np.sqrt(np.sum(penalty_square(ii))) for ii in mesh_xy]
# mesh_x = [ii[0] for ii in mesh_xy]
# mesh_y = [ii[1] for ii in mesh_xy]
# cmap = plt.cm.get_cmap('viridis')
# cmap2 = plt.cm.get_cmap('plasma')
# #scatter0 = plt.scatter(mesh_x,mesh_y,c=L_eps_vec,cmap = cmap,norm=colors.LogNorm())#vmin=1e29, vmax=6e29))#,zorder = 0)
# #scatter0 = plt.imshow(L_eps,extent =[vec_epsx[0], vec_epsx[-1], vec_epsy[0], vec_epsy[-1]],norm=colors.LogNorm())
# scatter0 = plt.scatter(mesh_x,mesh_y,c=L_eps_vec,cmap = cmap,norm=mcolors.LogNorm(),zorder = 0)
# plt.scatter(rel_err['eps'][0][0],rel_err['eps'][0][1],marker = 'x', c = 'black',zorder = 3, alpha = 1)
# scatter1= plt.scatter(iter_x,iter_y,zorder = 2,c=np.arange(len(iter_x)),cmap = cmap2)
# plt.plot(x_line,y_line,'-', c = 'cyan',zorder = 1, alpha = 0.5,dashes=(5, 10))
# plt.xlim(0.8*2.3e-6,1.2*2.3e-6)
# plt.ylim(0.8*2.3e-6,1.2*2.3e-6)
# # plt.xlim(0.98*rel_err['eps'][0],1.02*rel_err['eps'][0])
# # plt.ylim(0.98*rel_err['eps'][1],1.02*rel_err['eps'][1])
# scatter1= plt.scatter(iter_x,iter_y,zorder = 2,c=np.arange(len(iter_x)),cmap = cmap2)
# #plt.plot(iter_x,iter_y,zorder = 1,c = 'red')

# squarex = [0.8*rel_err['eps'][0][0],1.2*rel_err['eps'][0][0],1.2*rel_err['eps'][0][0],0.8*rel_err['eps'][0][0],0.8*rel_err['eps'][0][0]]
# squarey = [0.8*rel_err['eps'][0][1],0.8*rel_err['eps'][0][1],1.2*rel_err['eps'][0][1],1.2*rel_err['eps'][0][1],0.8*rel_err['eps'][0][1]]
# #plt.plot(squarex,squarey)
# plt.xlabel('epsilon1')
# plt.ylabel('epsilon2')
# plt.title('Penality of a vectorial function of 1 & 2')
# cbar = plt.colorbar(scatter0)
# cbar.ax.get_yaxis().labelpad = 15
# cbar.set_label( "Penalty values", rotation=270)
# cbar = plt.colorbar(scatter1)
# cbar.ax.get_yaxis().labelpad = 15
# cbar.set_label( "# iteration LS", rotation=270)
# #plt.savefig('rand_err_1par_eps12_e3_scan10_02beta.png')
# #plt.savefig('penalty_crossx_epsxy_lines.png')
# plt.savefig('penalty_cross_eps12_lines_difbeta.png')


# #%%

# # vec_epsy_xfix = np.linspace(1e-6,4e-6,100)
# # index_y_sim_fin = [np.abs(ii-iter_y[-1])<2e-14 for ii in iter_y]
# # iter_x_xfin = [iter_x[ii] for ii in np.where(index_y_sim_fin)[0]]
# # cmap = plt.cm.get_cmap('viridis')
# # Pen_eps_vec_x_fixed = [penalty_square([ii,rel_err['eps'][1]]) for ii in vec_epsy_xfix]
# # Pen_est_eps_vec_x_fixed = [penalty_square([ii,rel_err['est_eps'][1][1]]) for ii in vec_epsy_xfix]
# # plt.plot(vec_epsy_xfix,Pen_eps_vec_x_fixed, linewidth = 2, label = 'eps',zorder = 0,c = 'red')
# # plt.plot(vec_epsy_xfix,Pen_est_eps_vec_x_fixed, linewidth = 2, label = 'estimated eps',zorder = 0, c ='black')
# # Pen_iter_y = [penalty_square([ii,rel_err['eps'][1]]) for ii in iter_x_xfin]
# # Pen_est_iter_y = [penalty_square([ii,rel_err['est_eps'][1][1]]) for ii in iter_x_xfin]
# # plt.scatter(iter_x_xfin,Pen_iter_y,c=np.arange(len(iter_x_xfin)),cmap = cmap,zorder = 2)
# # plt.scatter(iter_x_xfin,Pen_est_iter_y,c=np.arange(len(iter_x_xfin)),cmap = cmap,zorder = 2)
# # plt.scatter(rel_err['eps'][0],penalty_square(rel_err['eps'])[0]+1e32,  marker = 'x', c = 'black',zorder = 3)
# # plt.plot(iter_x_xfin,Pen_iter_y,zorder = 1)
# # plt.xlabel('epsilony')
# # plt.yscale('log')
# # plt.ylabel('penalty function')
# # plt.legend()
# # plt.title('Penalty function over lines')
# # #plt.savefig('Penalty_epsyFix_rand_e3_scan10.png')

# #  #%%
# # vec_epsx_yfix = np.linspace(2e-6,2.6e-6,1000)
# # index_x_sim_fin = [np.abs(ii-iter_x[-1])<2e-9 for ii in iter_x]
# # iter_y_xfin = [iter_y[ii] for ii in np.where(index_x_sim_fin)[0]]
# # cmap = plt.cm.get_cmap('viridis')
# # Pen_eps_vec_x_fixed = [penalty_square([rel_err['est_eps'][1][0],ii]) for ii in vec_epsy_xfix]
# # plt.plot(vec_epsy_xfix,Pen_eps_vec_x_fixed, linewidth = 2, label = 'with random error 1e3',zorder = 0)
# # Pen_iter_y = [penalty_square([rel_err['est_eps'][1][0],ii]) for ii in iter_y_xfin]
# # plt.scatter(iter_y_xfin,Pen_iter_y,c=np.arange(len(iter_y_xfin)),cmap = cmap,zorder = 2)
# # plt.scatter(rel_err['est_eps'][1][1],penalty_square([rel_err['eps'][0],rel_err['est_eps'][1][1]]),c='black',marker = 'x',zorder = 3)
# # plt.plot(iter_y_xfin,Pen_iter_y,zorder = 1)
# # plt.xlabel('epsilony')
# # plt.ylabel('penalty function')
# # plt.title('Penalty function for epsx~2.49')
# # #plt.savefig('Penalty_epsxFix_rand_e3_scan20.png')
# # #%%

# # cmap = plt.cm.get_cmap('viridis')
# # L1_eps_vec_x_ = [np.abs(L_over_eps_xy(rel_err['eps'][0],ii,parameters_sym)-constants[0]) for ii in vec_epsy_xfix]
# # plt.plot(vec_epsy_xfix,L1_eps_vec_x_, linewidth = 2, label = 'with random error 1e3')
# # L1_iter_y = [np.abs(L_over_eps_xy(rel_err['eps'][0],ii,parameters_sym)-constants[0]) for ii in iter_y_xfin]
# # plt.scatter(iter_y_xfin,L1_iter_y,c=np.arange(len(iter_y_xfin)),cmap = cmap,zorder = 2)
# # plt.plot(iter_y_xfin,L1_iter_y,zorder = 1)
# # plt.xlabel('epsilony')
# # plt.ylabel('Comp1 Penalty function')
# # plt.title('Component 1 Penalty function for epsx~2.49')
# # #plt.savefig('Penalty1_epsxFix_rand_e3_scan20.png')
# # #%%

# # def penalty2_square(xy):
# #     penalty = 0
# #     keys = list(param['par'].keys())
# #     for i in range(len(constants)-1):
# #        penalty += (L_over_eps_xy(xy[0],xy[1],param['par'][keys[i]])-constants[i+1])**2

# #     return [penalty]


# # Pen2_eps_vec_x_fixed = [np.sqrt(np.sum(penalty2_square([rel_err['eps'][0],ii]))) for ii in vec_epsy_xfix]
# # plt.plot(vec_epsy_xfix,Pen2_eps_vec_x_fixed, linewidth = 2, label = 'with random error 1e3')
# # Pen2_iter_y = [np.sqrt(np.sum(penalty2_square([rel_err['eps'][0],ii]))) for ii in iter_y_xfin]
# # plt.scatter(iter_y_xfin,Pen2_iter_y,c=np.arange(len(iter_y_xfin)),cmap = cmap,zorder = 2)
# # plt.plot(iter_y_xfin,Pen2_iter_y,zorder = 1)
# # plt.xlabel('epsilony')
# # plt.ylabel('Comp2 Penalty function')
# # plt.title('Component 2 Penalty function for epsx~2.49')
# # #plt.savefig('Penalty2_epsxFix_rand_e3_scan20.png')
# # # # # %%

# # # %%



# # # # %%
# # # with open('output_zoom.txt') as f:
# # #     contents = f.read()
# # # fig1,ax1 = plt.subplots(1,2,figsize = (20,10))
# # # it_norm_x1 = [(eval(ii.split('\nObjective')[0])[0]/(2.3e-6) -1) for ii in contents.split('=============\nSolution: ')[1:]]
# # # it_norm_x2 = [(eval(ii.split('\nObjective')[0])[2]/(2.3e-6) -1) for ii in contents.split('=============\nSolution: ')[1:]]
# # # it_norm_y1 = [(eval(ii.split('\nObjective')[0])[1]/(2.3e-6) -1) for ii in contents.split('=============\nSolution: ')[1:]]
# # # it_norm_y2 = [(eval(ii.split('\nObjective')[0])[3]/(2.3e-6) -1) for ii in contents.split('=============\nSolution: ')[1:]]

# # # cmap = plt.cm.get_cmap('viridis')


# # # scatter0 = ax1[0].scatter(it_norm_x1[1:],it_norm_x2[1:],c=np.arange(len(it_norm_x1)-1),cmap = cmap)
# # # scatter1 = ax1[1].scatter(it_norm_y1[1:],it_norm_y2[1:],c=np.arange(len(it_norm_y1)-1), cmap = cmap)    
# # # ax1[0].scatter(0,0, c = 'red',s = 40)
# # # ax1[1].scatter(0,0, c = 'red',s = 40)
# # # ax1[0].scatter((epsx1/2.3e-6)-1,(epsx2/2.3e-6)-1,marker   = 'x', c = 'black',s = 100,linewidths=3,zorder = 3)
# # # ax1[1].scatter((epsy1/2.3e-6)-1,(epsy2/2.3e-6)-1,marker   = 'x', c = 'black',s = 100,
# # # linewidths=3)
# # # ax1[0].set_xlabel(r'$\frac{(\vec{\epsilon}_{x})_{1}-(\vec{\epsilon}_{x0})_{1}}{(\vec{\epsilon}_{x})_{1}}$', fontsize = 20)
# # # ax1[0].set_ylabel(r'$\frac{(\vec{\epsilon}_{x})_{2}-(\vec{\epsilon}_{x0})_{2}}{(\vec{\epsilon}_{x})_{2}}$', fontsize = 20)
# # # ax1[1].set_xlabel(r'$\frac{(\vec{\epsilon}_{y})_{1}-(\vec{\epsilon}_{y0})_{1}}{(\vec{\epsilon}_{y})_{1}}$', fontsize = 20)
# # # ax1[1].set_ylabel(r'$\frac{(\vec{\epsilon}_{y})_{2}-(\vec{\epsilon}_{y0})_{2}}{(\vec{\epsilon}_{y})_{2}}$', fontsize = 20)

# # # cbar1 = fig1.colorbar(scatter0, ax=ax1[0])
# # # cbar2 = fig1.colorbar(scatter1, ax=ax1[1])


# # # # %%
# # # iter_x1 = [eval(ii.split('\nObjective')[0])[0] for ii in contents.split('=============\nSolution: ')[1:]]
# # # iter_x2 = [eval(ii.split('\nObjective')[0])[2] for ii in contents.split('=============\nSolution: ')[1:]]
# # # iter_y1 = [eval(ii.split('\nObjective')[0])[1] for ii in contents.split('=============\nSolution: ')[1:]]
# # # iter_y2 = [eval(ii.split('\nObjective')[0])[3] for ii in contents.split('=============\nSolution: ')[1:]]

# # # L_target = lumi.L(epsx1,epsy1,epsx2,epsy2, *parameters_sym) 
# # # L_iteration = [lumi.L(iter_x1[j],iter_y1[j],iter_x2[j],iter_y2[j], *parameters_sym) for j in range(len(iter_x1))]

# # # rel_err_L = [abs(L_target-L_iteration[ii+len(L_iteration)-100])/L_target for ii in range(len(L_iteration)-(len(L_iteration)-100))]

# # # fig3,ax3 = plt.subplots(figsize = (20,10))
# # # fig3.suptitle('Relative error on Luminosity estimation over the last 100 iterations', fontsize=20)
# # # ax3.plot(np.linspace(len(L_iteration)-100,len(L_iteration)-1,len(L_iteration)-(len(L_iteration)-100)),rel_err_L)
# # # ax3.set_xlabel('Iterations of Least Squares', fontsize = 20)
# # # ax3.set_ylabel(r'$\frac{(\mathcal{L}_{target}-\mathcal{L}_{iter})}{\mathcal{L}_{target}}$', fontsize = 20)


# # # %%

# %%

# CHECK IF THE CHANGES IN THE PARAMETERS WOULD LEAD TO AN OFFSET IN THE VALUES OF THE FUNCTION OR THE STRUCTURE IS MORE COMPLEX, IN CASE WHAT IS THIS BEHAVIOUR?
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#NOISE CROSSING ANGLE!!!!!!!!!!!!!! MOVING ONLY ORBIT
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


parameters_sym_x = [f,nb,N,N,energy_tot,energy_tot,dif_mu0x,dif_mu0y,dif_px,dif_py, alphax,alphay,alphax,alphay,sigmaz,sigmaz,
            betax,betay,betax+1e-4,betay+1e-4,deltap_p0,deltap_p0,dmu0x,dmu0y,-dmu0x,-dmu0y,dpx,dpy,-dpx,-dpy]

parameters_sym_y = [f,nb,N,N,energy_tot,energy_tot,dif_mu0x,dif_mu0y,dif_py,dif_px, alphax,alphay,alphax,alphay,sigmaz,sigmaz,
            betax,betay,betax,betay,deltap_p0,deltap_p0,dmu0x,dmu0y,-dmu0x,-dmu0y,dpx,dpy,-dpx,-dpy]


parameters_sym_x2 = [f,nb,N,N,energy_tot,energy_tot,dif_mu0x,dif_mu0y,dif_px,dif_py, alphax,alphay,alphax,alphay,sigmaz,sigmaz,
             betax,betay,betax*0.98,betay*0.98,deltap_p0,deltap_p0,dmu0x,dmu0y,-dmu0x,-dmu0y,dpx,dpy,-dpx,-dpy]

parameters_sym_y2 = [f,nb,N,N,energy_tot,energy_tot,dif_mu0x,dif_mu0y,dif_py,dif_px, alphax,alphay,alphax,alphay,sigmaz,sigmaz,
             betax,betay,betax*0.98,betay*0.98,deltap_p0,deltap_p0,dmu0x,dmu0y,-dmu0x,-dmu0y,dpx,dpy,-dpx,-dpy]

# positions of the parameters!!!
dict_par = {'f':0,'nb':1,'N1':2,'N2':3,'energy_tot1':4,'energy_tot2':5,'mu0x':6,'mu0y':7,'px':8,'py':9,
                    'alphax':10,'alphay':11,'sigmaz':14,
                    'betax':16,'betay':17,'deltap_p0_1':20,'deltap_p0_2':21,
                    'dmu0x_1':22,'dmu0y_1':23,'dmu0x_2':24,'dmu0y_2':25,'dpx_1':26,'dpy_1':27,'dpx_2':28,'dpy_2':29}




#Luminosity that we imagine and the real one!
def L_over_parameters_sym(epsx1,epsy1,epsx2,epsy2, par = parameters_sym_x):
    '''
    Ridefinition of the luminosity model to set the nominal configuration as default one
    Input:
        - epsx1: emittance for beam1 on the x-axis
        - epsx2: emittance for beam2 on the x-axis
        - epsy1: emittance for beam1 on the y-axis
        - epsy2: emittance for beam2 on the y-axis
        - par:  machine parameters
                (default parameters_sym_1, the nominal configuration)

    Output:
        - the luminosity value with the chosen input.
    '''
    return lumi.L(epsx1,epsy1,epsx2,epsy2,*par)


# computation in order to obtain the shift in order to change the original luminosity of a percentage!
def percent_sym_short(epsx1,epsy1,epsx2,epsy2,delta,param,m):
    '''
    It returns the percentage of change from the nominal luminosity
    changing the parameter 'param' of 'delta' 
    Input:
        - epsx1: emittance for beam1 on the x-axis
        - epsx2: emittance for beam2 on the x-axis
        - epsy1: emittance for beam1 on the y-axis
        - epsy2: emittance for beam2 on the y-axis
        - delta: the shift value of the parameter 
        - param: the name of the parameter to shift
        - m: the sign of the shift in the luminosity
        - par:  machine parameters
                (default parameters_sym_1, the nominal configuration)

    Output:
        - the shift in percentage of the luminosity value
    '''
    delta_param = copy.copy(parameters_sym_x)
    delta_param[dict_par[param]] += delta
    
    # for symmetry reasons
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
    Inversion the luminosity function in order to obtain the emittances without considering 
    the luminosity measurement error, shifting the parameters in the dict_shift once a time.
    Input:
        - epsx1: emittance for beam1 on the x-axis
        - epsx2: emittance for beam2 on the x-axis
        - epsy1: emittance for beam1 on the y-axis
        - epsy2: emittance for beam2 on the y-axis
        - dict_shift: a dictionary in which the key is the name of the parameter
                      to change, while the value is a list containing how much should
                      be the shift in percentage on the nominal luminosity value
        - nfev: the maximum number of iteration of the non-linear LS
        - verbose: to print some output
        - par:  machine parameters
                (default parameters_sym_1, the nominal configuration)

    Output:
        - root.x: the numerical solution
        - root.fun: the output of the system at the numerical solution
        - root.jac: the Jacobian of the system at the numerical solution
        - time_LS: the computation time
        - root.nfev: the number of iteration of the non-linear LS
   
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
            ##!! this is for do combination of the shifts once at the time
            delta_par[f'{dict_shift[param][j]}{param}'] = copy.copy(parameters_sym_x)
            delta_par[f'{dict_shift[param][j]}{param}'][dict_par[param]]+= shift_par
            
            if param in ['betax','betay','alphax','alphay']:
                delta_par[f'{dict_shift[param][j]}{param}'][dict_par[param]+2]+= shift_par
            if param in ['sigmaz']:
                delta_par[f'{dict_shift[param][j]}{param}'][dict_par[param]+1]+= shift_par
            ##!! this is for do combination of the shifts all togheter, to get more combination
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
    par_inter = 'sigmaz'

    if par_inter in dict_shift:
        return [roots.x,roots.fun, roots.jac, time_LS, roots.nfev,[delta_par[f'{dict_shift[par_inter][j]}{par_inter}'][18:20] for j in range(len(dict_shift[par_inter]))]]
    else:
        return [roots.x,roots.fun, roots.jac, time_LS, roots.nfev]

########################################
def inv_gauss_single_par_randerr(epsx1,epsy1,epsx2,epsy2,dict_shift,nfev = 3000, verbose = 0, iteration  = 0):
    '''
    Inversion the luminosity function in order to obtain the emittances considering 
    the luminosity measurement error.
    It consider FOUR different initial configuration for the machine parameters,
    shifting both the choise using for each the shifts in the dict_shift once a time.
    Input:
        - epsx1: emittance for beam1 on the x-axis
        - epsx2: emittance for beam2 on the x-axis
        - epsy1: emittance for beam1 on the y-axis
        - epsy2: emittance for beam2 on the y-axis
        - dict_shift: a dictionary in which the key is the name of the parameter
                      to change, while the value is a list containing how much should
                      be the shift in percentage on the nominal luminosity value
        - nfev: the maximum number of iteration of the non-linear LS
        - verbose: to print some output


    Output:
        - root.x: the numerical solution
        - root.fun: the output of the system at the numerical solution
        - root.jac: the Jacobian of the system at the numerical solution
        - time_LS: the computation time
        - root.nfev: the number of iteration of the non-linear LS
   
    '''
    print("ua")
    iteration*=0
    n_shifts = 6
    delta_parx = {'': parameters_sym_x}
    delta_pary = {'': parameters_sym_y}
    delta_parx2 = {'': parameters_sym_x2}
    delta_pary2 = {'': parameters_sym_y2}
    index_par = 0
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
                    ##!! this is for do combination of the shifts all togheter, to get more combination
                    delta_parx[keys[p]+f'{dict_shift[param][j]}{param}_{k}'] = copy.copy(delta_parx[keys[p]])
                    delta_parx[keys[p]+f'{dict_shift[param][j]}{param}_{k}'][dict_par[param]]+= shift_par
                    delta_pary[keys[p]+f'{dict_shift[param][j]}{param}_{k}'] = copy.copy(delta_pary[keys[p]])
                    delta_pary[keys[p]+f'{dict_shift[param][j]}{param}_{k}'][dict_par[param]]+= shift_par
                    delta_parx2[keys[p]+f'{dict_shift[param][j]}{param}_{k}'] = copy.copy(delta_parx2[keys[p]])
                    delta_parx2[keys[p]+f'{dict_shift[param][j]}{param}_{k}'][dict_par[param]]+= shift_par
                    delta_pary2[keys[p]+f'{dict_shift[param][j]}{param}_{k}'] = copy.copy(delta_pary2[keys[p]])
                    delta_pary2[keys[p]+f'{dict_shift[param][j]}{param}_{k}'][dict_par[param]]+= shift_par
                    ##!! this is for do combination of the shifts once at the time
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

    
    print({'parx' :delta_parx})
    print({'pary' :delta_pary})
    print({'parx2' :delta_parx2})
    print({'pary2' :delta_pary2})
    
    # compute the luminosity in the different chosen initial parameters and add a noise
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

    
    # compute the luminosity but for all the shifts of the initial chosen parameters
    # and add a noise
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

    
    def func_to_zero(x):
        output = [L_over_parameters_sym(x[0],x[1],x[2],x[3],par= parameters_sym_x)-constants[0]]
        output = np.concatenate([output,[L_over_parameters_sym(x[0],x[1],x[2],x[3],par= parameters_sym_y)-constants[1]]])
        output = np.concatenate([output,[L_over_parameters_sym(x[0],x[1],x[2],x[3],par= parameters_sym_x2)-constants[2]]])
        output = np.concatenate([output,[L_over_parameters_sym(x[0],x[1],x[2],x[3],par= parameters_sym_y2)-constants[3]]])
        for i in range(len(delta_parx)-1):
            param = list(delta_parx.keys())[i+1]
            for index_par in range(4):
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

    return [roots.x,roots.fun, roots.jac, time_LS, roots.nfev]
    
########################################



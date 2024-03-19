#%%
import numpy as np
from numpy import linalg as LA
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import json
import itertools
from scipy.stats import gaussian_kde

n_jobs = 9
#n_generations = 2


#fig, axs = plt.subplots(n_jobs)
#if n_jobs !=1:
fig,ax = plt.subplots(5,2,figsize=(13,35))
titles = ['mu_p_b_a_z','mu_p_b_z','mu_p_a_z','mu_p_b_a','mu_b_a_z',\
          'p_b_a_z','mu_p_z','b_a_z','mu_p','b_a']
for jobs in range(n_jobs):
    jobs = jobs
    #print(f'{jobs:03}')
    path = f'000_child/{jobs:03}_child'
    js = open(path + '/output_data.json')
    data_out = json.load(js)
    # rel_err_x = np.append(0,[LA.norm([data_out['est_eps'][ii][jj]-data_out['eps'][ii][jj] for jj in [0,2]])\
    #                          /LA.norm(data_out['eps'][ii][0:4:2]) for ii in range(len(data_out['est_eps']))])
    # rel_err_y = np.append(0,[LA.norm([data_out['est_eps'][ii][jj]-data_out['eps'][ii][jj] for jj in [1,3]])/\
    #                          LA.norm(data_out['eps'][ii][1:5:2]) for ii in range(len(data_out['est_eps']))])
    rel_err_x = np.append(0,[LA.norm([data_out['est_eps'][ii][0]-data_out['eps'][ii][0]])/LA.norm(data_out['eps'][ii][0]) for ii in range(len(data_out['est_eps']))])
    rel_err_y = np.append(0,[LA.norm([data_out['est_eps'][ii][1]-data_out['eps'][ii][1]])/LA.norm(data_out['eps'][ii][1]) for ii in range(len(data_out['est_eps']))])
    eps = data_out['eps']
    length = len(rel_err_y)-1
    timing = [data_out['LS'][i][0] for i in range(length)]
    nfev = [data_out['LS'][i][1] for i in range(length)]
    av_time = np.sum(timing)/length
    av_nfev = np.sum([nfev])/length

    # xy = np.vstack([rel_err_x[1:],rel_err_y[1:]])
    # z = gaussian_kde(xy)(xy)    
    

    ax_x = int(jobs/2)
    ax_y = np.mod(jobs,2)

    #remove outliers
    ax[ax_x,ax_y].set_title(f'{titles[jobs]} ({av_time}s_{av_nfev})')
    ax[ax_x,ax_y].scatter(rel_err_x[0],rel_err_y[0], color ='green')
    scatter = ax[ax_x,ax_y].scatter(rel_err_x[1:],rel_err_y[1:], s=50)
    ax[ax_x,ax_y].set_xlabel('rel err on eps x')
    ax[ax_x,ax_y].set_ylabel('rel err on eps y')
    # cbar = plt.colorbar(scatter)
    # cbar.set_label('Density')


fig.savefig(f'LS.png')


#%%
from inversion_fun import Inv_gauss_xy12 as  inv_g
import random
import json
import contextlib
import numpy as np

dic6 = {'mu0x':[0.01],'mu0y':[0.01],'px':[0.01],'py':[0.01],'sigmaz':[0.01]}
dic8 = {'mu0x':[0.01],'mu0y':[0.01],'px':[0.01],'py':[0.01]}
LS = []
eps = []
sol_LS = {} #np.zeros([cfg['number_of_iterations'],4])
f_sol = {} #np.zeros([cfg['number_of_iterations'],len(dict_shift)+1])
sol = {}# Jac_sol = np.zeros([cfg['number_of_iterations'],len(dict_shift)+1,4])
# delta_sigmaz = {}
k = 0
for dict_shift in [dic6,dic8]:
    k+=1
    d = 4+2*k
    print(d)
    random.seed(42)
    jfile = open(f'outliers_{d}.json')
    data = json.load(jfile)
    for i in range(500):
        epsx1 = random.uniform(2.3e-6,2.6e-6)
        epsy1 = random.uniform(2.3e-6,2.6e-6)
        epsx2 = random.uniform(2.3e-6,2.6e-6)
        epsy2 = random.uniform(2.3e-6,2.6e-6)
        if i in [int(i.split('_')[0]) for i in list(data.keys())]:
            print(i)
            eps = [epsx1,epsy1,epsx2,epsy2]
            print(eps)
            assert eps == data[f'{i}_out']['eps']
            iterations = data[f'{i}_out']['LS'][1]
            sol_iter = np.zeros([iterations,4])
            f_iter = np.zeros([iterations, len(dict_shift)+1])
            # for j in range(iterations):
            #     print(j)
            with open(f"output_{d}_{eps}","w") as h,contextlib.redirect_stdout(h):
                inv_g.inv_gauss_single_par(epsx1,epsy1,epsx2,epsy2,dict_shift, verbose = 2)
                # if 'sigmaz' in dict_shift:
                #     [sol_iter[j],f_iter[j],Jac_sol,timing_LS,nfev_LS,delta_sigmaz] = inv_g.inv_gauss_single_par(epsx1,epsy1,epsx2,epsy2,dict_shift, verbose = 2)
                # else:
                #     [sol_iter[j],f_iter[j],Jac_sol,timing_LS,nfev_LS] = inv_g.inv_gauss_single_par(epsx1,epsy1,epsx2,epsy2,dict_shift,verbose = 2)
            
            
            # sol_LS[f'{d}_{i}_iter'] = sol_iter
            # f_sol[f'{d}_{i}_iter'] = f_iter
            # sol[f'{d}_{i}_iter'] = eps
        
        
    #     LS.append([timing_LS,nfev_LS])

    # sol_LS = [list(list(sol_LS)[ii]) for ii in range(len(sol_LS))]
    # f_sol = [list(list(f_sol)[ii]) for ii in range(len(f_sol))]
    # Jac_sol = [[list(list(Jac_sol)[ii][jj]) for jj in range(len(Jac_sol[ii]))] for ii in range(len(Jac_sol))]
    # if 'sigmaz' in dict_shift:
    #     rel_err = {'est_eps': sol_LS, 'eps': eps, 'f_sol': f_sol, 'Jacobian': Jac_sol, 'LS': LS, 'parameters_sigmaz': delta_sigmaz}
    # else:
    #     rel_err = {'est_eps': sol_LS, 'eps': eps, 'f_sol': f_sol, 'Jacobian': Jac_sol, 'LS': LS}

# %%
from inversion_fun import Inv_gauss_xy12 as  inv_g
dict_shift = {'mu0x':[0.01],'mu0y':[0.01],'px':[0.01],'py':[0.01],'sigmaz':[0.01]}

[epsx1,epsy1,epsx2,epsy2] = [2.343406394096152e-06, 2.338997639433031e-06, 2.375196259018437e-06, 2.352349136270418e-06]
sol_LS = inv_g.inv_gauss_single_par(epsx1,epsy1,epsx2,epsy2,dict_shift,verbose = 2)[0]
# %%
#TREES PLOT FOR 4 DIMENSIONS problems
from matplotlib import pyplot as plt
from inversion_fun import Inv_gauss_xy12 as  inv_g
from inversion_fun import lumi_formula_no_inner as lumi
import random
import json
import contextlib
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec
import numpy as np

[f,nb,N,energy_tot] = [11245,2736,1.4e11,6800]
[dif_mu0x,dif_mu0y] = [0,0]
[dif_px,dif_py] = [0,320e-6]#[160e-6,160e-6]
[alphax,alphay] = [0,0]
[sigmaz] = [0.35]
[betax,betay] =[0.3,0.3]
[deltap_p0] = [0]
[dmu0x,dmu0y] = [0,0]
[dpx,dpy] = [0,0]
parameters_sym = [f,nb,N,N,energy_tot,energy_tot,dif_mu0x,dif_mu0y,dif_px,dif_py, alphax,alphay,alphax,alphay,sigmaz,sigmaz,
            betax,betay,betax,betay,deltap_p0,deltap_p0,dmu0x,dmu0y,-dmu0x,-dmu0y,dpx,dpy,-dpx,-dpy]


names = 'mu_cross'
folder = '../examples/trees/plots'
dict = {'mu0x':[0.01],'mu0y':[0.01],'px':[0.01],'py':[0.01]}
titles = r'Reconstruction of target $\mathbf{\vec{\epsilon}}$ by Luminosity, varying separation & crossing angle ($\mathbf{\vec{\mu}}$,$\mathbf{\vec{\theta}})$'
titles_iter = r'Reconstruction iterations in order to reach a specific $\vec{\epsilon}$, varying $\mathbf{\mu}$ & $\mathbf{\theta}$'
titles_iter_zoom = r'Zoom on reconstruction iterations in order to reach a specific $\vec{\epsilon}$, varying $\mathbf{\mu}$ & $\mathbf{\theta}$'
path = f'../examples/trees/tree_noerrxy12/000'
js = open(path + '/output_data.json')
data_out = json.load(js)

frac_int_eps = [[data_out['eps'][ii][jj]/(2.3e-6)-1 for jj in\
                  range(len(data_out['eps'][ii]))] for ii in range(len(data_out['eps']))]
frac_init_est_eps = [[data_out['est_eps'][ii][jj]/(2.3e-6)-1 for jj in\
                  range(len(data_out['est_eps'][ii]))] for ii in range(len(data_out['est_eps']))]
rel_err_noerrxy12 = [np.linalg.norm([(data_out['eps'][ii][jj]-data_out['est_eps'][ii][jj]) for jj in range(len(data_out['eps'][ii]))])/np.linalg.norm(data_out['eps'][ii]) for ii in range(len(data_out['eps']))]
rel_err1_noerrxy12 = [[(data_out['eps'][ii][jj]-data_out['est_eps'][ii][jj])/data_out['eps'][ii][jj] for jj in range(len(data_out['eps'][ii]))] for ii in range(len(data_out['eps']))]
for jobs in range(3):
    # fig,ax = plt.subplots(1,2,figsize = (20,10))
    jobs = jobs

    #square definition
    x = [-0.2, 0.2, 0.2, -0.2, -0.2]
    y = [-0.2, -0.2, 0.2, 0.2, -0.2]

    path = f'../examples/trees/tree_noerrxy12/{jobs+1:03}'
    js = open(path + '/output_data.json')
    data_out = json.load(js)
    frac_int_eps = np.concatenate([frac_int_eps,[[data_out['eps'][ii][jj]/(2.3e-6)-1 for jj in\
                  range(len(data_out['eps'][ii]))] for ii in range(len(data_out['eps']))]],axis = 0)
    frac_init_est_eps = np.concatenate([frac_init_est_eps,[[data_out['est_eps'][ii][jj]/(2.3e-6)-1 for jj in\
                  range(len(data_out['est_eps'][ii]))] for ii in range(len(data_out['est_eps']))]],axis = 0)

    rel_err_noerrxy12 = np.append(rel_err_noerrxy12,[np.linalg.norm([(data_out['eps'][ii][jj]-data_out['est_eps'][ii][jj]) for jj in range(len(data_out['eps'][ii]))])/np.linalg.norm(data_out['eps'][ii]) for ii in range(len(data_out['eps']))])
    rel_err1_noerrxy12 = np.append(rel_err1_noerrxy12,[[(data_out['eps'][ii][jj]-data_out['est_eps'][ii][jj])/data_out['eps'][ii][jj] for jj in range(len(data_out['eps'][ii]))] for ii in range(len(data_out['eps']))])
    

frac_int_eps_x1 = [frac_int_eps[ii][0] for ii in range(len(frac_int_eps))]
frac_int_eps_x2 = [frac_int_eps[ii][2] for ii in range(len(frac_int_eps))]
frac_int_eps_y1 = [frac_int_eps[ii][1] for ii in range(len(frac_int_eps))]
frac_int_eps_y2 = [frac_int_eps[ii][3] for ii in range(len(frac_int_eps))]
frac_init_est_eps_x1 = [frac_init_est_eps[ii][0] for ii in range(len(frac_init_est_eps))]
frac_init_est_eps_x2 = [frac_init_est_eps[ii][2] for ii in range(len(frac_init_est_eps))]
frac_init_est_eps_y1 = [frac_init_est_eps[ii][1] for ii in range(len(frac_init_est_eps))]
frac_init_est_eps_y2 = [frac_init_est_eps[ii][3] for ii in range(len(frac_init_est_eps))]

    # fig.suptitle(titles[jobs], fontsize=20)

    # Plot the square
#     ax[0].plot(x, y, linewidth = 4)
#     ax[1].plot(x, y, linewidth = 4)

#     ax[0].scatter(frac_int_eps_x1[0],frac_int_eps_x2[0],marker   = 'x', c = 'black',s = 100,linewidths=3, label = r'target $\mathbf{\vec{\epsilon}_{x}}$')
#     ax[1].scatter(frac_int_eps_y1[0],frac_int_eps_y2[0],marker   = 'x', c = 'black',s = 100,
#     linewidths=3, label = r'target $\mathbf{\vec{\epsilon}_{y}})$')
#     ax[0].scatter(frac_init_est_eps_x1[0],frac_init_est_eps_x2[0], c = 'orange', s = 30,label = r'estimated $\mathbf{\vec{\epsilon}_{x}}$')
#     ax[1].scatter(frac_init_est_eps_y1[0],frac_init_est_eps_y2[0], c = 'orange', s = 30,label = r'estimated $\mathbf{\vec{\epsilon}_{y}}$')
#     ax[0].scatter(0,0, c = 'red',s = 40, label = r'intial guess ($\mathbf{\vec{\epsilon}_{x0}}$)')
#     ax[1].scatter(0,0, c = 'red',s = 40, label = r'initial guess ($\mathbf{\vec{\epsilon}_{y0}}$')
#     ax[0].set_xlabel(r'$\frac{(\vec{\epsilon}_{x})_{1}-(\vec{\epsilon}_{x0})_{1}}{(\vec{\epsilon}_{x})_{1}}$', fontsize = 20)
#     ax[0].set_ylabel(r'$\frac{(\vec{\epsilon}_{x})_{2}-(\vec{\epsilon}_{x0})_{2}}{(\vec{\epsilon}_{x})_{2}}$', fontsize = 20)
#     ax[1].set_xlabel(r'$\frac{(\vec{\epsilon}_{y})_{1}-(\vec{\epsilon}_{y0})_{1}}{(\vec{\epsilon}_{y})_{1}}$', fontsize = 20)
#     ax[1].set_ylabel(r'$\frac{(\vec{\epsilon}_{y})_{2}-(\vec{\epsilon}_{y0})_{2}}{(\vec{\epsilon}_{y})_{2}}$', fontsize = 20)

#     ax[0].legend(fontsize = 20, loc = 'upper right')
#     ax[1].legend(fontsize = 20, loc = 'upper right')
#     ax[0].set_xlim((-0.25,0.25))
#     ax[0].set_ylim((-0.25,0.25))
#     ax[1].set_xlim((-0.25,0.25))
#     ax[1].set_ylim((-0.25,0.25))
#     # fig.savefig(f'{folder}/one_eps/{names[jobs]}_LS.png')

# ########## COMPUTATION ITERATIONS

#     random.seed(42)
#     epsx1 = random.uniform(2.3e-6-4.6e-7,2.3e-6+4.6e-7)
#     epsy1 = random.uniform(2.3e-6-4.6e-7,2.3e-6+4.6e-7)
#     epsx2 = random.uniform(2.3e-6-4.6e-7,2.3e-6+4.6e-7)
#     epsy2 = random.uniform(2.3e-6-4.6e-7,2.3e-6+4.6e-7)
#     eps = [epsx1,epsy1,epsx2,epsy2]


#     dict_shift = dict[jobs]
#     # for j in range(iterations):
#     #     print(j)
#     with open(f"output_zoom.txt","w") as h,contextlib.redirect_stdout(h):
#         [est_eps,function, jacobian, time_LS, iterations] = inv_g.inv_gauss_single_par(epsx1,epsy1,epsx2,epsy2,dict_shift, verbose = 2, iteration = 0)




#     with open('output_zoom.txt') as f:
#         contents = f.read()
#     from scipy.stats import gaussian_kde
#     fig1,ax1 = plt.subplots(1,2,figsize = (20,10))
#     it_norm_x1 = [(eval(ii.split('\nObjective')[0])[0]/(2.3e-6) -1) for ii in contents.split('=============\nSolution: ')[1:]]
#     it_norm_x2 = [(eval(ii.split('\nObjective')[0])[2]/(2.3e-6) -1) for ii in contents.split('=============\nSolution: ')[1:]]
#     it_norm_y1 = [(eval(ii.split('\nObjective')[0])[1]/(2.3e-6) -1) for ii in contents.split('=============\nSolution: ')[1:]]
#     it_norm_y2 = [(eval(ii.split('\nObjective')[0])[3]/(2.3e-6) -1) for ii in contents.split('=============\nSolution: ')[1:]]

#     cmap = plt.cm.get_cmap('viridis')

    
#     scatter0 = ax1[0].scatter(it_norm_x1[1:],it_norm_x2[1:],c=np.arange(len(it_norm_x1)-1),cmap = cmap)
#     scatter1 = ax1[1].scatter(it_norm_y1[1:],it_norm_y2[1:],c=np.arange(len(it_norm_y1)-1), cmap = cmap)    
#     ax1[0].scatter(0,0, c = 'red',s = 40)
#     ax1[1].scatter(0,0, c = 'red',s = 40)
#     ax1[0].scatter(frac_int_eps_x1[0],frac_int_eps_x2[0],marker   = 'x', c = 'black',s = 100,linewidths=3,zorder = 3)
#     ax1[1].scatter(frac_int_eps_y1[0],frac_int_eps_y2[0],marker   = 'x', c = 'black',s = 100,
#     linewidths=3)
#     fig1.suptitle(titles_iter[jobs], fontsize=20)
#     ax1[0].set_xlabel(r'$\frac{(\vec{\epsilon}_{x})_{1}-(\vec{\epsilon}_{x0})_{1}}{(\vec{\epsilon}_{x})_{1}}$', fontsize = 20)
#     ax1[0].set_ylabel(r'$\frac{(\vec{\epsilon}_{x})_{2}-(\vec{\epsilon}_{x0})_{2}}{(\vec{\epsilon}_{x})_{2}}$', fontsize = 20)
#     ax1[1].set_xlabel(r'$\frac{(\vec{\epsilon}_{y})_{1}-(\vec{\epsilon}_{y0})_{1}}{(\vec{\epsilon}_{y})_{1}}$', fontsize = 20)
#     ax1[1].set_ylabel(r'$\frac{(\vec{\epsilon}_{y})_{2}-(\vec{\epsilon}_{y0})_{2}}{(\vec{\epsilon}_{y})_{2}}$', fontsize = 20)
#     ax1[0].set_xlim((-0.25,0.25))
#     ax1[0].set_ylim((-0.25,0.25))
#     ax1[1].set_xlim((-0.25,0.25))
#     ax1[1].set_ylim((-0.25,0.25))
#     cbar1 = fig1.colorbar(scatter0, ax=ax1[0])
#     cbar2 = fig1.colorbar(scatter1, ax=ax1[1])
#     # fig1.savefig(f'{folder}/one_eps/{names[jobs]}_iter_LS.png')

# ########################

#     fig2,ax2 = plt.subplots(1,2,figsize = (20,10))
#     ax2[0].plot(it_norm_x1,it_norm_x2, zorder=1)
#     ax2[1].plot(it_norm_y1,it_norm_y2, zorder=1)
#     scatter0 = ax2[0].scatter(it_norm_x1[1:],it_norm_x2[1:],c=np.arange(len(it_norm_x1)-1),cmap = cmap, zorder=2)
#     scatter1 = ax2[1].scatter(it_norm_y1[1:],it_norm_y2[1:],c=np.arange(len(it_norm_y1)-1), cmap = cmap, zorder=2)    
#     ax2[0].scatter(0,0, c = 'red',s = 40)
#     ax2[1].scatter(0,0, c = 'red',s = 40)
#     ax2[0].scatter(frac_int_eps_x1[0],frac_int_eps_x2[0],marker   = 'x', c = 'black',s = 100,linewidths=3,zorder = 3)
#     ax2[1].scatter(frac_int_eps_y1[0],frac_int_eps_y2[0],marker   = 'x', c = 'black',s = 100,linewidths=3,zorder = 3)

#     fig2.suptitle(titles_iter_zoom[jobs], fontsize=20)
#     ax2[0].set_xlabel(r'$\frac{(\vec{\epsilon}_{x})_{1}-(\vec{\epsilon}_{x0})_{1}}{(\vec{\epsilon}_{x})_{1}}$', fontsize = 20)
#     ax2[0].set_ylabel(r'$\frac{(\vec{\epsilon}_{x})_{2}-(\vec{\epsilon}_{x0})_{2}}{(\vec{\epsilon}_{x})_{2}}$', fontsize = 20)
#     ax2[1].set_xlabel(r'$\frac{(\vec{\epsilon}_{y})_{1}-(\vec{\epsilon}_{y0})_{1}}{(\vec{\epsilon}_{y})_{1}}$', fontsize = 20)
#     ax2[1].set_ylabel(r'$\frac{(\vec{\epsilon}_{y})_{2}-(\vec{\epsilon}_{y0})_{2}}{(\vec{\epsilon}_{y})_{2}}$', fontsize = 20)
    

#     cbar1 = fig2.colorbar(scatter0, ax=ax2[0])
#     cbar2 = fig2.colorbar(scatter1, ax=ax2[1])
#     # fig2.savefig(f'{folder}/one_eps/zoom_{names[jobs]}_iter_LS.png')
# ################

#     iter_x1 = [eval(ii.split('\nObjective')[0])[0] for ii in contents.split('=============\nSolution: ')[1:]]
#     iter_x2 = [eval(ii.split('\nObjective')[0])[2] for ii in contents.split('=============\nSolution: ')[1:]]
#     iter_y1 = [eval(ii.split('\nObjective')[0])[1] for ii in contents.split('=============\nSolution: ')[1:]]
#     iter_y2 = [eval(ii.split('\nObjective')[0])[3] for ii in contents.split('=============\nSolution: ')[1:]]

#     L_target = lumi.L(epsx1,epsy1,epsx2,epsy2, *parameters_sym) 
#     L_iteration = [lumi.L(iter_x1[j],iter_y1[j],iter_x2[j],iter_y2[j], *parameters_sym) for j in range(len(iter_x1))]
    
#     rel_err_L = [abs(L_target-L_iteration[ii+len(L_iteration)-100])/L_target for ii in range(len(L_iteration)-(len(L_iteration)-100))]
    
#     fig3,ax3 = plt.subplots(figsize = (20,10))
#     fig3.suptitle('Relative error on Luminosity estimation over the last 100 iterations', fontsize=20)
#     ax3.plot(np.linspace(len(L_iteration)-100,len(L_iteration)-1,len(L_iteration)-(len(L_iteration)-100)),rel_err_L)
#     ax3.set_xlabel('Iterations of Least Squares', fontsize = 20)
#     ax3.set_ylabel(r'$\frac{(\mathcal{L}_{target}-\mathcal{L}_{iter})}{\mathcal{L}_{target}}$', fontsize = 20)
#     # fig3.savefig(f'{folder}/one_eps/rel_err_L_{names[jobs]}.png')

# #################

#     fig4,ax4 = plt.subplots(1,2,figsize = (20,10))
#     fig4.suptitle(titles[jobs], fontsize=20)


#     # Plot the square
#     ax4[0].plot(x, y, linewidth = 4)
#     ax4[1].plot(x, y, linewidth = 4)

#     ax4[0].scatter(frac_int_eps_x1,frac_int_eps_x2,marker   = 'x', c = 'black',s = 100,linewidths=3, label = r'target $\mathbf{\vec{\epsilon}_{x}}$')
#     ax4[1].scatter(frac_int_eps_y1,frac_int_eps_y2,marker   = 'x', c = 'black',s = 100,
#     linewidths=3, label = r'target $\mathbf{\vec{\epsilon}_{y}})$')
#     ax4[0].scatter(frac_init_est_eps_x1,frac_init_est_eps_x2, c = 'orange', s = 10,label = r'estimated $\mathbf{\vec{\epsilon}_{x}}$')
#     ax4[1].scatter(frac_init_est_eps_y1,frac_init_est_eps_y2, c = 'orange', s = 10,label = r'estimated $\mathbf{\vec{\epsilon}_{y}}$')
#     ax4[0].scatter(0,0, c = 'red',s = 40, label = r'intial guess ($\mathbf{\vec{\epsilon}_{x0}}$)')
#     ax4[1].scatter(0,0, c = 'red',s = 40, label = r'initial guess ($\mathbf{\vec{\epsilon}_{y0}}$')
#     ax4[0].set_xlabel(r'$\frac{(\vec{\epsilon}_{x})_{1}-(\vec{\epsilon}_{x0})_{1}}{(\vec{\epsilon}_{x})_{1}}$', fontsize = 20)
#     ax4[0].set_ylabel(r'$\frac{(\vec{\epsilon}_{x})_{2}-(\vec{\epsilon}_{x0})_{2}}{(\vec{\epsilon}_{x})_{2}}$', fontsize = 20)
#     ax4[1].set_xlabel(r'$\frac{(\vec{\epsilon}_{y})_{1}-(\vec{\epsilon}_{y0})_{1}}{(\vec{\epsilon}_{y})_{1}}$', fontsize = 20)
#     ax4[1].set_ylabel(r'$\frac{(\vec{\epsilon}_{y})_{2}-(\vec{\epsilon}_{y0})_{2}}{(\vec{\epsilon}_{y})_{2}}$', fontsize = 20)

#     ax4[0].legend(fontsize = 20, loc = 'upper right')
#     ax4[1].legend(fontsize = 20, loc = 'upper right')
#     # fig4.savefig(f'{folder}/all_{names[jobs]}_LS.png')


#%%
#TREES PLOT FOR 2 DIMENSIONS problems

from matplotlib import pyplot as plt
from inversion_fun import Inv_gauss_12 as  inv_g
from inversion_fun import lumi_formula_no_inner as lumi
import random
import json
import contextlib
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec
import numpy as np

names = ['mux','muy','thetax','thetay']
folder = 'results_alpha0_crossy_eps12_difbet/home/mrufolo/storage/Inverting_luminosity/inv_gauss_tree_maker/000_errxy12_2a'
[f,nb,N,energy_tot] = [11245,2736,1.4e11,6800]
[dif_mu0x,dif_mu0y] = [0,0]
[dif_px,dif_py] = [0,320e-6]#[160e-6,160e-6]
[alphax,alphay] = [0,0]
[sigmaz] = [0.35]
[betax,betay] =[0.3,0.3]
[deltap_p0] = [0]
[dmu0x,dmu0y] = [0,0]
[dpx,dpy] = [0,0]
parameters_sym = [f,nb,N,N,energy_tot,energy_tot,dif_mu0x,dif_mu0y,dif_px,dif_py, alphax,alphay,alphax,alphay,sigmaz,sigmaz,
            betax,betay,betax+1e-4,betay+1e-4,deltap_p0,deltap_p0,dmu0x,dmu0y,-dmu0x,-dmu0y,dpx,dpy,-dpx,-dpy]

dict = [{'mu0x':[0.01]},{'mu0y':[0.01]},{'px':[0.01]},{'py':[0.01]}]
titles = [r'Reconstruction of target $\mathbf{(\epsilon_{1},\epsilon_{2})}$ by Luminosity, varying the offset on y ($\mathbf{\mu_{x}}$)',r'Reconstruction of target $\mathbf{(\epsilon_{1},\epsilon_{2})}$ by Luminosity, varying the offset on y ($\mathbf{mu_{y}}$)',r'Reconstruction of target $\mathbf{(\epsilon_{1},\epsilon_{2})}$ by Luminosity, varying crossing angle on x ($\mathbf{\theta_{x}})$',r'Reconstruction of target $\mathbf{(\epsilon_{1},\epsilon_{2})}$ by Luminosity, varying the crossing angle on y ($\mathbf{\theta_{y}}$)' ]
titles_iter = [r'Reconstruction iterations in order to reach a specific $\mathbf{(\epsilon_{1},\epsilon_{2})}$, varying $\mathbf{\mu_{x}}$',r'Reconstruction iterations in order to reach a specific $\mathbf{(\epsilon_{1},\epsilon_{2})}$, varying $\mathbf{mu_{y}}$',r'Reconstruction iterations in order to reach a specific $\mathbf{(\epsilon_{1},\epsilon_{2})}$, varying $\mathbf{\theta_{x}}$',r'Reconstruction iterations in order to reach a specific $\mathbf{(\epsilon_{1},\epsilon_{2})}$, varying $\mathbf{\theta_{y}}$']
titles_iter_zoom = [r'Zoom on reconstruction iterations in order to reach a specific $\mathbf{(\epsilon_{1},\epsilon_{2})}$, varying $\mathbf{\mu_{x}}$',r'Zoom on reconstruction iterations in order to reach a specific $\mathbf{(\epsilon_{1},\epsilon_{2})}$, varying $\mathbf{mu_{y}}$',r'Zoom on reconstruction iterations in order to reach a specific $\mathbf{(\epsilon_{1},\epsilon_{2})}$, varying $\mathbf{\theta_{x}}$',r'Zoom on reconstruction iterations in order to reach a specific $\mathbf{(\epsilon_{1},\epsilon_{2})}$, varying $\mathbf{\theta_{y}}$']
rel_err_errxy = []
rel_err1_errxy  = []
for jobs in range(4):
    fig,ax = plt.subplots()
    jobs = jobs

    #square definition
    x = [-0.2, 0.2, 0.2, -0.2, -0.2]
    y = [-0.2, -0.2, 0.2, 0.2, -0.2]

    # path = f'000_child/{jobs:03}_child'
    path = f'000_errxy/{jobs:03}'
    js = open(path + '/output_data.json')
    data_out = json.load(js)
    if len(data_out['eps'][0]) == 4:
        data_out['eps'] = [i[1:3] for i in data_out['eps']]
    frac_int_eps = [[data_out['eps'][ii][jj]/(2.3e-6)-1 for jj in\
                  range(len(data_out['eps'][ii]))] for ii in range(len(data_out['eps']))]
    frac_init_est_eps = [[data_out['est_eps'][ii][jj]/(2.3e-6)-1 for jj in\
                  range(len(data_out['est_eps'][ii]))] for ii in range(len(data_out['est_eps']))]
    frac_int_eps_1 = [frac_int_eps[ii][0] for ii in range(len(frac_int_eps))]
    frac_int_eps_2 = [frac_int_eps[ii][1] for ii in range(len(frac_int_eps))]
    frac_init_est_eps_1 = [frac_init_est_eps[ii][0] for ii in range(len(frac_init_est_eps))]
    frac_init_est_eps_2 = [frac_init_est_eps[ii][1] for ii in range(len(frac_init_est_eps))]
    

    fig.suptitle(titles[jobs], fontsize=20)

    # Plot the square
    ax.plot(x, y, linewidth = 4)

    ax.scatter(frac_int_eps_1[0],frac_int_eps_2[0],marker   = 'x', c = 'black',s = 100,linewidths=3, label = r'target $\mathbf{\vec{\epsilon}}$')
    ax.scatter(frac_init_est_eps_1[0],frac_init_est_eps_2[0], c = 'orange', s = 30,label = r'estimated $\mathbf{\vec{\epsilon}}$')
    ax.scatter(0,0, c = 'red',s = 40, label = r'intial guess ($\mathbf{\vec{\epsilon}_{0}}$)')
    ax.set_xlabel(r'$\frac{(\vec{\epsilon}_{1})-(\vec{\epsilon}_{10})}{(\vec{\epsilon}_{1})}$', fontsize = 20)
    ax.set_ylabel(r'$\frac{(\vec{\epsilon}_{2})-(\vec{\epsilon}_{20})}{(\vec{\epsilon}_{2})}$', fontsize = 20)
    
    ax.legend(fontsize = 14, loc = 'upper right')
    ax.set_xlim((-0.25,0.25))
    ax.set_ylim((-0.25,0.25))
    # fig.savefig(f'{folder}/one_eps/{names[jobs]}_LS.png' ,bbox_inches = "tight")

########## COMPUTATION ITERATIONS

    random.seed(42)
    eps1 = random.uniform(2.3e-6-4.6e-7,2.3e-6+4.6e-7)
    eps2 = random.uniform(2.3e-6-4.6e-7,2.3e-6+4.6e-7)
    eps = [eps1,eps1,eps2,eps2]


    dict_shift = dict[jobs]
    # for j in range(iterations):
    #     print(j)
    with open(f"output_zoom.txt","w") as h,contextlib.redirect_stdout(h):
        [est_eps,function, jacobian, time_LS, iterations] = inv_g.inv_gauss_12(eps1,eps2,dict_shift, verbose = 2)




    with open('output_zoom.txt') as f:
        contents = f.read()
        
    fig1,ax1 = plt.subplots()
    it_norm_1 = [(eval(ii.split('\nObjective')[0])[0]/(2.3e-6) -1) for ii in contents.split('=============\nSolution: ')[1:]]
    it_norm_2 = [(eval(ii.split('\nObjective')[0])[1]/(2.3e-6) -1) for ii in contents.split('=============\nSolution: ')[1:]]
    
    cmap = plt.cm.get_cmap('viridis')

    
    scatter0 = ax1.scatter(it_norm_1[1:],it_norm_2[1:],c=np.arange(len(it_norm_1)-1),cmap = cmap)
    ax1.scatter(0,0, c = 'red',s = 40)
    ax1.scatter(frac_int_eps_1[0],frac_int_eps_2[0],marker   = 'x', c = 'black',s = 100,linewidths=3,zorder = 3)
    fig1.suptitle(titles_iter[jobs], fontsize=20)
    ax1.set_xlabel(r'$\frac{(\vec{\epsilon}_{1})-(\vec{\epsilon}_{10})}{(\vec{\epsilon}_{1})}$', fontsize = 20)
    ax1.set_ylabel(r'$\frac{(\vec{\epsilon}_{2})-(\vec{\epsilon}_{20})}{(\vec{\epsilon}_{2})}$', fontsize = 20)
    ax1.set_xlim((-0.25,0.25))
    ax1.set_ylim((-0.25,0.25))
    cbar1 = fig1.colorbar(scatter0, ax=ax1)
    # fig1.savefig(f'{folder}/one_eps/{names[jobs]}_iter_LS.png' ,bbox_inches = "tight")

########################

    fig2,ax2 = plt.subplots()
    ax2.plot(it_norm_1,it_norm_2, zorder=1)
    scatter0 = ax2.scatter(it_norm_1[1:],it_norm_2[1:],c=np.arange(len(it_norm_1)-1),cmap = cmap, zorder=2)
    ax2.scatter(0,0, c = 'red',s = 40)
    ax2.scatter(frac_int_eps_1[0],frac_int_eps_2[0],marker   = 'x', c = 'black',s = 100,linewidths=3,zorder = 3)
    
    fig2.suptitle(titles_iter_zoom[jobs], fontsize=20)
    ax2.set_xlabel(r'$\frac{(\vec{\epsilon}_{1})-(\vec{\epsilon}_{10})}{(\vec{\epsilon}_{1})}$', fontsize = 20)
    ax2.set_ylabel(r'$\frac{(\vec{\epsilon}_{2})-(\vec{\epsilon}_{20})}{(\vec{\epsilon}_{2})}$', fontsize = 20)
    
    cbar1 = fig1.colorbar(scatter0, ax=ax2)
    # fig2.savefig(f'{folder}/one_eps/zoom_{names[jobs]}_iter_LS.png' ,bbox_inches = "tight")
################

    iter_1 = [eval(ii.split('\nObjective')[0])[0] for ii in contents.split('=============\nSolution: ')[1:]]
    iter_2 = [eval(ii.split('\nObjective')[0])[1] for ii in contents.split('=============\nSolution: ')[1:]]
    
    L_target = lumi.L(eps1,eps1,eps2,eps2, *parameters_sym) 
    L_iteration = [lumi.L(iter_1[j],iter_2[j],iter_1[j],iter_2[j], *parameters_sym) for j in range(len(iter_1))]
    
    rel_err_L = [abs(L_target-L_iteration[ii+len(L_iteration)-20])/L_target for ii in range(len(L_iteration)-(len(L_iteration)-20))]
    
    fig3,ax3 = plt.subplots()
    fig3.suptitle('Relative error on Luminosity estimation over the last 20 iterations', fontsize=20)
    ax3.plot(np.linspace(len(L_iteration)-20,len(L_iteration)-1,len(L_iteration)-(len(L_iteration)-20)),rel_err_L)
    ax3.set_xlabel('Iterations of Least Squares', fontsize = 10)
    ax3.set_ylabel(r'$\frac{(\mathcal{L}_{target}-\mathcal{L}_{iter})}{\mathcal{L}_{target}}$', fontsize = 20)
    # fig3.savefig(f'{folder}/one_eps/rel_err_L_{names[jobs]}.png' ,bbox_inches = "tight")


#################

    fig4,ax4 = plt.subplots()
    fig4.suptitle(titles[jobs], fontsize=20)


    # Plot the square
    ax4.plot(x, y, linewidth = 4)
    
    ax4.scatter(frac_int_eps_1,frac_int_eps_2,marker   = 'x', c = 'black',s = 100,linewidths=3, label = r'target $\mathbf{\vec{\epsilon}}$')
    ax4.scatter(frac_init_est_eps_1,frac_init_est_eps_2, c = 'orange', s = 10,label = r'estimated $\mathbf{\vec{\epsilon}}$')
    ax4.scatter(0,0, c = 'red',s = 40, label = r'intial guess ($\mathbf{\vec{\epsilon}_{0}}$)')
    ax4.set_xlabel(r'$\frac{(\vec{\epsilon}_{1})-(\vec{\epsilon}_{10})}{(\vec{\epsilon}_{1})}$', fontsize = 20)
    ax4.set_ylabel(r'$\frac{(\vec{\epsilon}_{2})-(\vec{\epsilon}_{20})}{(\vec{\epsilon}_{2})}$', fontsize = 20)
    
    ax4.legend(fontsize = 14, loc = 'upper right')
    # fig4.savefig(f'{folder}/all_{names[jobs]}_LS.png' ,bbox_inches = "tight")
#     # Plot the square

    
    # rel_err_noerrxy = np.append(rel_err_noerrxy,[np.linalg.norm([(data_out['eps'][ii][jj]-data_out['est_eps'][ii][jj]) for jj in range(len(data_out['eps'][ii]))])/np.linalg.norm(data_out['eps'][ii]) for ii in range(len(data_out['eps']))])
    # rel_err1_noerrxy = np.append(rel_err1_noerrxy,[[(data_out['eps'][ii][jj]-data_out['est_eps'][ii][jj])/data_out['eps'][ii][jj] for jj in range(len(data_out['eps'][ii]))] for ii in range(len(data_out['eps']))])
    # rel_err_errxy = np.append(rel_err_errxy,[np.linalg.norm([(data_out['eps'][ii][jj]-data_out['est_eps'][ii][int(jj/2)]) for jj in range(len(data_out['eps'][ii]))])/np.linalg.norm(data_out['eps'][ii]) for ii in range(len(data_out['eps']))])
    # rel_err1_errxy = np.append(rel_err1_errxy,[[(data_out['eps'][ii][jj]-data_out['est_eps'][ii][int(jj/2)])/data_out['eps'][ii][jj] for jj in range(len(data_out['eps'][ii]))] for ii in range(len(data_out['eps']))])
    rel_err_errxy = np.append(rel_err_errxy,[np.linalg.norm([(data_out['eps'][ii][jj]-data_out['est_eps'][ii][int(jj)]) for jj in range(len(data_out['eps'][ii]))])/np.linalg.norm(data_out['eps'][ii]) for ii in range(len(data_out['eps']))])
    rel_err1_errxy = np.append(rel_err1_errxy,[[(data_out['eps'][ii][jj]-data_out['est_eps'][ii][int(jj)])/data_out['eps'][ii][jj] for jj in range(len(data_out['eps'][ii]))] for ii in range(len(data_out['eps']))])
    
# plt.figure()
# plt.hist(rel_err,bins = 'auto')
# plt.title('Histogram of the L2 norm of the relative error')
# plt.xlabel('L2 norm of relative error')
# plt.ylabel('Recurrence')
# rel_err1_noerrxy = reject_outliers(rel_err1_noerrxy)
# rel_err1_errxy = reject_outliers(rel_err1_errxy)
# rel_err1_errxy2 = reject_outliers(rel_err1_errxy2)

# plt.figure()

# plt.hist(rel_err1_errxy, bins = 200, alpha = 0.5, color = 'red', label = 'with error')
# plt.hist(rel_err1_errxy2, bins = 8, alpha = 0.5, color = 'green', label = 'with error')
# plt.hist(rel_err1_noerrxy, bins =200, color = 'black', label = 'without error')
# plt.xscale('log')
# plt.title(r'Component-wise histogram of $\Delta{\vec{\epsilon}}/\vec{\epsilon}$ in log scale', fontsize = 16)
# plt.legend()
# plt.xticks(fontsize = 13)
# plt.yticks(fontsize = 13)
# plt.xlabel('Relative error', fontsize = 16)
# plt.ylabel('Recurrence', fontsize = 16)
# plt.savefig('paper_hist/hist_xy_noerr_err.png')
    



# %%

from Inverting_luminosity import Inv_gauss_xy12 as  inv_g
import random
import json
import contextlib
import numpy as np

#dic6 = {'mu0x':[0.01],'mu0y':[0.01],'px':[0.01],'py':[0.01],'sigmaz':[0.01]}
dict_shift = {'mu0x':[0.01],'mu0y':[0.01],'px':[0.01],'py':[0.01]}
LS = []
eps = []
sol_LS = {} #np.zeros([cfg['number_of_iterations'],4])
f_sol = {} #np.zeros([cfg['number_of_iterations'],len(dict_shift)+1])
sol = {}# Jac_sol = np.zeros([cfg['number_of_iterations'],len(dict_shift)+1,4])
# delta_sigmaz = {}


random.seed(42)
epsx1 = random.uniform(2.3e-6,2.6e-6)
epsy1 = random.uniform(2.3e-6,2.6e-6)
epsx2 = random.uniform(2.3e-6,2.6e-6)
epsy2 = random.uniform(2.3e-6,2.6e-6)

eps = [epsx1,epsy1,epsx2,epsy2]
print(eps)
# for j in range(iterations):
#     print(j)
with open(f"output_zoom.txt","w") as h,contextlib.redirect_stdout(h):
    [est_eps,function, jacobian, time_LS, iterations] = inv_g.inv_gauss_single_par(epsx1,epsy1,epsx2,epsy2,dict_shift, verbose = 2, iteration = 0)

frac_est_zoom_x1 = est_eps[0]/(2.3e-6)
frac_est_zoom_x2  = est_eps[2]/(2.3e-6)
frac_est_zoom_y1 = est_eps[1]/(2.3e-6)
frac_est_zoom_y2  = est_eps[3]/(2.3e-6)
ax[0].scatter(frac_est_zoom_x1,frac_est_zoom_x2,marker   = 'x', c = 'cyan',s = 100,linewidths=3, label = r'real $\vec{\epsilon_x}$')
ax[1].scatter(frac_est_zoom_y1,frac_est_zoom_y2,marker   = 'x', c = 'cyan',s = 100,
linewidths=3, label = r'real $\vec{\epsilon_y}$')

# %%

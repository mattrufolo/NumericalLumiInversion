#%%
import numpy as np
from numpy import linalg as LA
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import json
import itertools
from scipy.stats import gaussian_kde
from inversion_fun import Inv_gauss_xy as  inv_g
from inversion_fun import Inv_gauss_xy12 as  inv_g_xy12
from inversion_fun import lumi_formula_no_inner as lumi
import random
import contextlib
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec

def reject_outliers(data, m=7):
    idx = abs(data - np.mean(data)) < m * np.std(data)
    return data[idx], idx

#%%
#TREES PLOT FOR 2 DIMENSIONS problems without error



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


names = 'mux_noerrxy'
folder = '../examples/trees/plots/noerrxy'
dict = {'mu0x':[0.01]}

titles = r'Reconstruction of target $\mathbf{(\epsilon_{1},\epsilon_{2})}$ by Luminosity, varying the offset on y ($\mathbf{\mu_{x}}$)'
titles_iter = r'Reconstruction iterations in order to reach a specific $\mathbf{(\epsilon_{1},\epsilon_{2})}$, varying $\mathbf{\mu_{x}}$'
titles_iter_zoom = r'Zoom on reconstruction iterations in order to reach a specific $\mathbf{(\epsilon_{1},\epsilon_{2})}$, varying $\mathbf{\mu_{x}}$'
path = f'../examples/trees/tree_noerrxy/000'
js = open(path + '/output_data.json')
data_out = json.load(js)

if len(data_out['eps'][0]) == 4:
    data_out['eps'] = [i[1:3] for i in data_out['eps']]

frac_int_eps = [[data_out['eps'][ii][jj]/(2.3e-6)-1 for jj in\
                range(len(data_out['eps'][ii]))] for ii in range(len(data_out['eps']))]
frac_init_est_eps = [[data_out['est_eps'][ii][jj]/(2.3e-6)-1 for jj in\
                range(len(data_out['est_eps'][ii]))] for ii in range(len(data_out['est_eps']))]
rel_err_noerrxy = [np.linalg.norm([(data_out['eps'][ii][jj]-data_out['est_eps'][ii][int(jj)]) for jj in range(len(data_out['eps'][ii]))])/np.linalg.norm(data_out['eps'][ii]) for ii in range(len(data_out['eps']))]
rel_err1_noerrxy  = [[(data_out['eps'][ii][jj]-data_out['est_eps'][ii][int(jj)])/data_out['eps'][ii][jj] for jj in range(len(data_out['eps'][ii]))] for ii in range(len(data_out['eps']))]

for jobs in range(3):
    #square definition



    path = f'../examples/trees/tree_noerrxy/{jobs+1:03}'
    js = open(path + '/output_data.json')
    data_out = json.load(js)
    if len(data_out['eps'][0]) == 4:
        data_out['eps'] = [i[1:3] for i in data_out['eps']]
    frac_int_eps = np.concatenate([frac_int_eps,[[data_out['eps'][ii][jj]/(2.3e-6)-1 for jj in\
                range(len(data_out['eps'][ii]))] for ii in range(len(data_out['eps']))]],axis = 0)
    frac_init_est_eps = np.concatenate([frac_init_est_eps,[[data_out['est_eps'][ii][jj]/(2.3e-6)-1 for jj in\
                range(len(data_out['est_eps'][ii]))] for ii in range(len(data_out['est_eps']))]],axis = 0)
    
    rel_err_noerrxy = np.append(rel_err_noerrxy,[np.linalg.norm([(data_out['eps'][ii][jj]-data_out['est_eps'][ii][int(jj)]) for jj in range(len(data_out['eps'][ii]))])/np.linalg.norm(data_out['eps'][ii]) for ii in range(len(data_out['eps']))])
    rel_err1_noerrxy = np.append(rel_err1_noerrxy,[[(data_out['eps'][ii][jj]-data_out['est_eps'][ii][int(jj)])/data_out['eps'][ii][jj] for jj in range(len(data_out['eps'][ii]))] for ii in range(len(data_out['eps']))])
    

rel_err_noerrxy, indeces = reject_outliers(rel_err_noerrxy)
rel_err1_noerrxy = np.array(rel_err1_noerrxy)[indeces.repeat(2)]
frac_int_eps_1 = np.array([frac_int_eps[ii][0] for ii in range(len(frac_int_eps))])[indeces]
frac_int_eps_2 = np.array([frac_int_eps[ii][1] for ii in range(len(frac_int_eps))])[indeces]
frac_init_est_eps_1 = np.array([frac_init_est_eps[ii][0] for ii in range(len(frac_init_est_eps))])[indeces]
frac_init_est_eps_2 = np.array([frac_init_est_eps[ii][1] for ii in range(len(frac_init_est_eps))])[indeces]

fig,ax = plt.subplots()

fig.suptitle(titles, fontsize=20)

x = [-0.2, 0.2, 0.2, -0.2, -0.2]
y = [-0.2, -0.2, 0.2, 0.2, -0.2]

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
fig.savefig(f'{folder}/one_eps/{names}_LS.png' ,bbox_inches = "tight")

########## COMPUTATION ITERATIONS

random.seed(42)
eps1 = random.uniform(2.3e-6-4.6e-7,2.3e-6+4.6e-7)
eps2 = random.uniform(2.3e-6-4.6e-7,2.3e-6+4.6e-7)
eps = [eps1,eps1,eps2,eps2]


dict_shift = dict
with open(f"output_zoom.txt","w") as h,contextlib.redirect_stdout(h):
    [est_eps,function, jacobian, time_LS, iterations] = inv_g.inv_gauss_xy(eps1,eps2,dict_shift, verbose = 2)




with open('output_zoom.txt') as f:
    contents = f.read()
    
fig1,ax1 = plt.subplots()
it_norm_1 = [(eval(ii.split('\nObjective')[0])[0]/(2.3e-6) -1) for ii in contents.split('=============\nSolution: ')[1:]]
it_norm_2 = [(eval(ii.split('\nObjective')[0])[1]/(2.3e-6) -1) for ii in contents.split('=============\nSolution: ')[1:]]

cmap = plt.cm.get_cmap('viridis')


scatter0 = ax1.scatter(it_norm_1[1:],it_norm_2[1:],c=np.arange(len(it_norm_1)-1),cmap = cmap)
ax1.scatter(0,0, c = 'red',s = 40)
ax1.scatter(frac_int_eps_1[0],frac_int_eps_2[0],marker   = 'x', c = 'black',s = 100,linewidths=3,zorder = 3)
fig1.suptitle(titles_iter, fontsize=20)
ax1.set_xlabel(r'$\frac{(\vec{\epsilon}_{1})-(\vec{\epsilon}_{10})}{(\vec{\epsilon}_{1})}$', fontsize = 20)
ax1.set_ylabel(r'$\frac{(\vec{\epsilon}_{2})-(\vec{\epsilon}_{20})}{(\vec{\epsilon}_{2})}$', fontsize = 20)
ax1.set_xlim((-0.25,0.25))
ax1.set_ylim((-0.25,0.25))
cbar1 = fig1.colorbar(scatter0, ax=ax1)
fig1.savefig(f'{folder}/one_eps/{names}_iter_LS.png' ,bbox_inches = "tight")

########################

fig2,ax2 = plt.subplots()
ax2.plot(it_norm_1,it_norm_2, zorder=1)
scatter0 = ax2.scatter(it_norm_1[1:],it_norm_2[1:],c=np.arange(len(it_norm_1)-1),cmap = cmap, zorder=2)
ax2.scatter(0,0, c = 'red',s = 40)
ax2.scatter(frac_int_eps_1[0],frac_int_eps_2[0],marker   = 'x', c = 'black',s = 100,linewidths=3,zorder = 3)

fig2.suptitle(titles_iter_zoom, fontsize=20)
ax2.set_xlabel(r'$\frac{(\vec{\epsilon}_{1})-(\vec{\epsilon}_{10})}{(\vec{\epsilon}_{1})}$', fontsize = 20)
ax2.set_ylabel(r'$\frac{(\vec{\epsilon}_{2})-(\vec{\epsilon}_{20})}{(\vec{\epsilon}_{2})}$', fontsize = 20)

cbar1 = fig1.colorbar(scatter0, ax=ax2)
fig2.savefig(f'{folder}/one_eps/zoom_{names}_iter_LS.png' ,bbox_inches = "tight")
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
fig3.savefig(f'{folder}/one_eps/rel_err_L_{names}.png' ,bbox_inches = "tight")


#################

fig4,ax4 = plt.subplots()
fig4.suptitle(titles, fontsize=20)


# Plot the square
ax4.plot(x, y, linewidth = 4)

ax4.scatter(frac_int_eps_1,frac_int_eps_2,marker   = 'x', c = 'black',s = 100,linewidths=3, label = r'target $\mathbf{\vec{\epsilon}}$')
ax4.scatter(frac_init_est_eps_1,frac_init_est_eps_2, c = 'orange', s = 10,label = r'estimated $\mathbf{\vec{\epsilon}}$')
ax4.scatter(0,0, c = 'red',s = 40, label = r'intial guess ($\mathbf{\vec{\epsilon}_{0}}$)')
ax4.set_xlabel(r'$\frac{(\vec{\epsilon}_{1})-(\vec{\epsilon}_{10})}{(\vec{\epsilon}_{1})}$', fontsize = 20)
ax4.set_ylabel(r'$\frac{(\vec{\epsilon}_{2})-(\vec{\epsilon}_{20})}{(\vec{\epsilon}_{2})}$', fontsize = 20)

ax4.legend(fontsize = 14, loc = 'upper right')
fig4.savefig(f'{folder}/all_{names}_LS.png' ,bbox_inches = "tight")
#     # Plot the square

plt.figure()

plt.hist(rel_err1_noerrxy, bins = 'auto')
plt.title(r'Component-wise histogram of $\Delta{\vec{\epsilon}}/\vec{\epsilon}$')
plt.xlabel('Relative error')
plt.ylabel('Recurrence')
plt.savefig(f'{folder}/histogram_rel_err.png')



#%%

#TREES PLOT FOR 2 DIMENSIONS problems with error, using 1IP


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


names = 'mux_errxy_1IP'
folder = '../examples/trees/plots/errxy_1IP'
dict = {'mu0x':[0.01]}

titles = r'Reconstruction of target $\mathbf{(\epsilon_{1},\epsilon_{2})}$ by Luminosity, varying the offset on y ($\mathbf{\mu_{x}}$)'
titles_iter = r'Reconstruction iterations in order to reach a specific $\mathbf{(\epsilon_{1},\epsilon_{2})}$, varying $\mathbf{\mu_{x}}$'
titles_iter_zoom = r'Zoom on reconstruction iterations in order to reach a specific $\mathbf{(\epsilon_{1},\epsilon_{2})}$, varying $\mathbf{\mu_{x}}$'
path = f'../examples/trees/tree_errxy_1IP/000'
js = open(path + '/output_data.json')
data_out = json.load(js)

if len(data_out['eps'][0]) == 4:
    data_out['eps'] = [i[1:3] for i in data_out['eps']]

frac_int_eps = [[data_out['eps'][ii][jj]/(2.3e-6)-1 for jj in\
                range(len(data_out['eps'][ii]))] for ii in range(len(data_out['eps']))]
frac_init_est_eps = [[data_out['est_eps'][ii][jj]/(2.3e-6)-1 for jj in\
                range(len(data_out['est_eps'][ii]))] for ii in range(len(data_out['est_eps']))]
rel_err_errxy = [np.linalg.norm([(data_out['eps'][ii][jj]-data_out['est_eps'][ii][int(jj)]) for jj in range(len(data_out['eps'][ii]))])/np.linalg.norm(data_out['eps'][ii]) for ii in range(len(data_out['eps']))]
rel_err1_errxy  = [[(data_out['eps'][ii][jj]-data_out['est_eps'][ii][int(jj)])/data_out['eps'][ii][jj] for jj in range(len(data_out['eps'][ii]))] for ii in range(len(data_out['eps']))]

for jobs in range(3):
    #square definition



    path = f'../examples/trees/tree_errxy_1IP/{jobs+1:03}'
    js = open(path + '/output_data.json')
    data_out = json.load(js)
    if len(data_out['eps'][0]) == 4:
        data_out['eps'] = [i[1:3] for i in data_out['eps']]
    frac_int_eps = np.concatenate([frac_int_eps,[[data_out['eps'][ii][jj]/(2.3e-6)-1 for jj in\
                range(len(data_out['eps'][ii]))] for ii in range(len(data_out['eps']))]],axis = 0)
    frac_init_est_eps = np.concatenate([frac_init_est_eps,[[data_out['est_eps'][ii][jj]/(2.3e-6)-1 for jj in\
                range(len(data_out['est_eps'][ii]))] for ii in range(len(data_out['est_eps']))]],axis = 0)
    
    rel_err_errxy = np.append(rel_err_errxy,[np.linalg.norm([(data_out['eps'][ii][jj]-data_out['est_eps'][ii][int(jj)]) for jj in range(len(data_out['eps'][ii]))])/np.linalg.norm(data_out['eps'][ii]) for ii in range(len(data_out['eps']))])
    rel_err1_errxy = np.append(rel_err1_errxy,[[(data_out['eps'][ii][jj]-data_out['est_eps'][ii][int(jj)])/data_out['eps'][ii][jj] for jj in range(len(data_out['eps'][ii]))] for ii in range(len(data_out['eps']))])
    

rel_err_errxy, indeces = reject_outliers(rel_err_errxy)
rel_err1_errxy = np.array(rel_err1_errxy)[indeces.repeat(2)]
frac_int_eps_1 = np.array([frac_int_eps[ii][0] for ii in range(len(frac_int_eps))])[indeces]
frac_int_eps_2 = np.array([frac_int_eps[ii][1] for ii in range(len(frac_int_eps))])[indeces]
frac_init_est_eps_1 = np.array([frac_init_est_eps[ii][0] for ii in range(len(frac_init_est_eps))])[indeces]
frac_init_est_eps_2 = np.array([frac_init_est_eps[ii][1] for ii in range(len(frac_init_est_eps))])[indeces]

fig,ax = plt.subplots()

fig.suptitle(titles, fontsize=20)

x = [-0.2, 0.2, 0.2, -0.2, -0.2]
y = [-0.2, -0.2, 0.2, 0.2, -0.2]

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
fig.savefig(f'{folder}/one_eps/{names}_LS.png' ,bbox_inches = "tight")

########## COMPUTATION ITERATIONS

random.seed(42)
eps1 = random.uniform(2.3e-6-4.6e-7,2.3e-6+4.6e-7)
eps2 = random.uniform(2.3e-6-4.6e-7,2.3e-6+4.6e-7)
eps = [eps1,eps1,eps2,eps2]


dict_shift = dict
with open(f"output_zoom.txt","w") as h,contextlib.redirect_stdout(h):
    [est_eps,function, jacobian, time_LS, iterations] = inv_g.inv_gauss_xy_randerr(eps1,eps2,dict_shift, verbose = 2)




with open('output_zoom.txt') as f:
    contents = f.read()
    
fig1,ax1 = plt.subplots()
it_norm_1 = [(eval(ii.split('\nObjective')[0])[0]/(2.3e-6) -1) for ii in contents.split('=============\nSolution: ')[1:]]
it_norm_2 = [(eval(ii.split('\nObjective')[0])[1]/(2.3e-6) -1) for ii in contents.split('=============\nSolution: ')[1:]]

cmap = plt.cm.get_cmap('viridis')


scatter0 = ax1.scatter(it_norm_1[1:],it_norm_2[1:],c=np.arange(len(it_norm_1)-1),cmap = cmap)
ax1.scatter(0,0, c = 'red',s = 40)
ax1.scatter(frac_int_eps_1[0],frac_int_eps_2[0],marker   = 'x', c = 'black',s = 100,linewidths=3,zorder = 3)
fig1.suptitle(titles_iter, fontsize=20)
ax1.set_xlabel(r'$\frac{(\vec{\epsilon}_{1})-(\vec{\epsilon}_{10})}{(\vec{\epsilon}_{1})}$', fontsize = 20)
ax1.set_ylabel(r'$\frac{(\vec{\epsilon}_{2})-(\vec{\epsilon}_{20})}{(\vec{\epsilon}_{2})}$', fontsize = 20)
ax1.set_xlim((-0.25,0.25))
ax1.set_ylim((-0.25,0.25))
cbar1 = fig1.colorbar(scatter0, ax=ax1)
fig1.savefig(f'{folder}/one_eps/{names}_iter_LS.png' ,bbox_inches = "tight")

########################

fig2,ax2 = plt.subplots()
ax2.plot(it_norm_1,it_norm_2, zorder=1)
scatter0 = ax2.scatter(it_norm_1[1:],it_norm_2[1:],c=np.arange(len(it_norm_1)-1),cmap = cmap, zorder=2)
ax2.scatter(0,0, c = 'red',s = 40)
ax2.scatter(frac_int_eps_1[0],frac_int_eps_2[0],marker   = 'x', c = 'black',s = 100,linewidths=3,zorder = 3)

fig2.suptitle(titles_iter_zoom, fontsize=20)
ax2.set_xlabel(r'$\frac{(\vec{\epsilon}_{1})-(\vec{\epsilon}_{10})}{(\vec{\epsilon}_{1})}$', fontsize = 20)
ax2.set_ylabel(r'$\frac{(\vec{\epsilon}_{2})-(\vec{\epsilon}_{20})}{(\vec{\epsilon}_{2})}$', fontsize = 20)

cbar1 = fig1.colorbar(scatter0, ax=ax2)
fig2.savefig(f'{folder}/one_eps/zoom_{names}_iter_LS.png' ,bbox_inches = "tight")
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
fig3.savefig(f'{folder}/one_eps/rel_err_L_{names}.png' ,bbox_inches = "tight")


#################

fig4,ax4 = plt.subplots()
fig4.suptitle(titles, fontsize=20)


# Plot the square
ax4.plot(x, y, linewidth = 4)

ax4.scatter(frac_int_eps_1,frac_int_eps_2,marker   = 'x', c = 'black',s = 100,linewidths=3, label = r'target $\mathbf{\vec{\epsilon}}$')
ax4.scatter(frac_init_est_eps_1,frac_init_est_eps_2, c = 'orange', s =60,label = r'estimated $\mathbf{\vec{\epsilon}}$')
ax4.scatter(0,0, c = 'red',s = 40, label = r'intial guess ($\mathbf{\vec{\epsilon}_{0}}$)')
ax4.set_xlabel(r'$\frac{(\vec{\epsilon}_{1})-(\vec{\epsilon}_{10})}{(\vec{\epsilon}_{1})}$', fontsize = 20)
ax4.set_ylabel(r'$\frac{(\vec{\epsilon}_{2})-(\vec{\epsilon}_{20})}{(\vec{\epsilon}_{2})}$', fontsize = 20)

ax4.legend(fontsize = 14, loc = 'upper right')
fig4.savefig(f'{folder}/all_{names}_LS.png' ,bbox_inches = "tight")
#     # Plot the square

plt.figure()

plt.hist(rel_err1_errxy, bins = 'auto')
plt.title(r'Component-wise histogram of $\Delta{\vec{\epsilon}}/\vec{\epsilon}$')
plt.xlabel('Relative error')
plt.ylabel('Recurrence')
plt.savefig(f'{folder}/histogram_rel_err.png')

#%%
#TREES PLOT FOR 2 DIMENSIONS problems with error, using 2IPs

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


names = 'mux_errxy_2IPS'
folder = '../examples/trees/plots/errxy_2IPS'
dict = {'mu0x':[0.01]}

titles = r'Reconstruction of target $\mathbf{(\epsilon_{1},\epsilon_{2})}$ by Luminosity, varying the offset on y ($\mathbf{\mu_{x}}$)'
titles_iter = r'Reconstruction iterations in order to reach a specific $\mathbf{(\epsilon_{1},\epsilon_{2})}$, varying $\mathbf{\mu_{x}}$'
titles_iter_zoom = r'Zoom on reconstruction iterations in order to reach a specific $\mathbf{(\epsilon_{1},\epsilon_{2})}$, varying $\mathbf{\mu_{x}}$'
path = f'../examples/trees/tree_errxy_2IPS/000'
js = open(path + '/output_data.json')
data_out = json.load(js)

if len(data_out['eps'][0]) == 4:
    data_out['eps'] = [i[1:3] for i in data_out['eps']]

frac_int_eps = [[data_out['eps'][ii][jj]/(2.3e-6)-1 for jj in\
                range(len(data_out['eps'][ii]))] for ii in range(len(data_out['eps']))]
frac_init_est_eps = [[data_out['est_eps'][ii][jj]/(2.3e-6)-1 for jj in\
                range(len(data_out['est_eps'][ii]))] for ii in range(len(data_out['est_eps']))]
rel_err_errxy_2IP = [np.linalg.norm([(data_out['eps'][ii][jj]-data_out['est_eps'][ii][int(jj)]) for jj in range(len(data_out['eps'][ii]))])/np.linalg.norm(data_out['eps'][ii]) for ii in range(len(data_out['eps']))]
rel_err1_errxy_2IP  = [[(data_out['eps'][ii][jj]-data_out['est_eps'][ii][int(jj)])/data_out['eps'][ii][jj] for jj in range(len(data_out['eps'][ii]))] for ii in range(len(data_out['eps']))]

for jobs in range(3):
    #square definition



    path = f'../examples/trees/tree_errxy_2IPS/{jobs+1:03}'
    js = open(path + '/output_data.json')
    data_out = json.load(js)
    if len(data_out['eps'][0]) == 4:
        data_out['eps'] = [i[1:3] for i in data_out['eps']]
    frac_int_eps = np.concatenate([frac_int_eps,[[data_out['eps'][ii][jj]/(2.3e-6)-1 for jj in\
                range(len(data_out['eps'][ii]))] for ii in range(len(data_out['eps']))]],axis = 0)
    frac_init_est_eps = np.concatenate([frac_init_est_eps,[[data_out['est_eps'][ii][jj]/(2.3e-6)-1 for jj in\
                range(len(data_out['est_eps'][ii]))] for ii in range(len(data_out['est_eps']))]],axis = 0)
    
    rel_err_errxy_2IP = np.append(rel_err_errxy_2IP,[np.linalg.norm([(data_out['eps'][ii][jj]-data_out['est_eps'][ii][int(jj)]) for jj in range(len(data_out['eps'][ii]))])/np.linalg.norm(data_out['eps'][ii]) for ii in range(len(data_out['eps']))])
    rel_err1_errxy_2IP = np.append(rel_err1_errxy_2IP,[[(data_out['eps'][ii][jj]-data_out['est_eps'][ii][int(jj)])/data_out['eps'][ii][jj] for jj in range(len(data_out['eps'][ii]))] for ii in range(len(data_out['eps']))])
    

rel_err_errxy_2IP, indeces = reject_outliers(rel_err_errxy_2IP)
rel_err1_errxy_2IP = np.array(rel_err1_errxy_2IP)[indeces.repeat(2)]
frac_int_eps_1 = np.array([frac_int_eps[ii][0] for ii in range(len(frac_int_eps))])[indeces]
frac_int_eps_2 = np.array([frac_int_eps[ii][1] for ii in range(len(frac_int_eps))])[indeces]
frac_init_est_eps_1 = np.array([frac_init_est_eps[ii][0] for ii in range(len(frac_init_est_eps))])[indeces]
frac_init_est_eps_2 = np.array([frac_init_est_eps[ii][1] for ii in range(len(frac_init_est_eps))])[indeces]

fig,ax = plt.subplots()

fig.suptitle(titles, fontsize=20)

x = [-0.2, 0.2, 0.2, -0.2, -0.2]
y = [-0.2, -0.2, 0.2, 0.2, -0.2]

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
fig.savefig(f'{folder}/one_eps/{names}_LS.png' ,bbox_inches = "tight")

########## COMPUTATION ITERATIONS

random.seed(42)
eps1 = random.uniform(2.3e-6-4.6e-7,2.3e-6+4.6e-7)
eps2 = random.uniform(2.3e-6-4.6e-7,2.3e-6+4.6e-7)
eps = [eps1,eps1,eps2,eps2]


dict_shift = dict
with open(f"output_zoom.txt","w") as h,contextlib.redirect_stdout(h):
    [est_eps,function, jacobian, time_LS, iterations] = inv_g.inv_gauss_xy_randerr_2(eps1,eps2,dict_shift, verbose = 2)




with open('output_zoom.txt') as f:
    contents = f.read()
    
fig1,ax1 = plt.subplots()
it_norm_1 = [(eval(ii.split('\nObjective')[0])[0]/(2.3e-6) -1) for ii in contents.split('=============\nSolution: ')[1:]]
it_norm_2 = [(eval(ii.split('\nObjective')[0])[1]/(2.3e-6) -1) for ii in contents.split('=============\nSolution: ')[1:]]

cmap = plt.cm.get_cmap('viridis')


scatter0 = ax1.scatter(it_norm_1[1:],it_norm_2[1:],c=np.arange(len(it_norm_1)-1),cmap = cmap)
ax1.scatter(0,0, c = 'red',s = 40)
ax1.scatter(frac_int_eps_1[0],frac_int_eps_2[0],marker   = 'x', c = 'black',s = 100,linewidths=3,zorder = 3)
fig1.suptitle(titles_iter, fontsize=20)
ax1.set_xlabel(r'$\frac{(\vec{\epsilon}_{1})-(\vec{\epsilon}_{10})}{(\vec{\epsilon}_{1})}$', fontsize = 20)
ax1.set_ylabel(r'$\frac{(\vec{\epsilon}_{2})-(\vec{\epsilon}_{20})}{(\vec{\epsilon}_{2})}$', fontsize = 20)
ax1.set_xlim((-0.25,0.25))
ax1.set_ylim((-0.25,0.25))
cbar1 = fig1.colorbar(scatter0, ax=ax1)
fig1.savefig(f'{folder}/one_eps/{names}_iter_LS.png' ,bbox_inches = "tight")

########################

fig2,ax2 = plt.subplots()
ax2.plot(it_norm_1,it_norm_2, zorder=1)
scatter0 = ax2.scatter(it_norm_1[1:],it_norm_2[1:],c=np.arange(len(it_norm_1)-1),cmap = cmap, zorder=2)
ax2.scatter(0,0, c = 'red',s = 40)
ax2.scatter(frac_int_eps_1[0],frac_int_eps_2[0],marker   = 'x', c = 'black',s = 100,linewidths=3,zorder = 3)

fig2.suptitle(titles_iter_zoom, fontsize=20)
ax2.set_xlabel(r'$\frac{(\vec{\epsilon}_{1})-(\vec{\epsilon}_{10})}{(\vec{\epsilon}_{1})}$', fontsize = 20)
ax2.set_ylabel(r'$\frac{(\vec{\epsilon}_{2})-(\vec{\epsilon}_{20})}{(\vec{\epsilon}_{2})}$', fontsize = 20)

cbar1 = fig1.colorbar(scatter0, ax=ax2)
fig2.savefig(f'{folder}/one_eps/zoom_{names}_iter_LS.png' ,bbox_inches = "tight")
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
fig3.savefig(f'{folder}/one_eps/rel_err_L_{names}.png' ,bbox_inches = "tight")


#################

fig4,ax4 = plt.subplots()
fig4.suptitle(titles, fontsize=20)


# Plot the square
ax4.plot(x, y, linewidth = 4)

ax4.scatter(frac_int_eps_1,frac_int_eps_2,marker   = 'x', c = 'black',s = 100,linewidths=3, label = r'target $\mathbf{\vec{\epsilon}}$')
ax4.scatter(frac_init_est_eps_1,frac_init_est_eps_2, c = 'orange', s =14,label = r'estimated $\mathbf{\vec{\epsilon}}$')
ax4.scatter(0,0, c = 'red',s = 40, label = r'intial guess ($\mathbf{\vec{\epsilon}_{0}}$)')
ax4.set_xlabel(r'$\frac{(\vec{\epsilon}_{1})-(\vec{\epsilon}_{10})}{(\vec{\epsilon}_{1})}$', fontsize = 20)
ax4.set_ylabel(r'$\frac{(\vec{\epsilon}_{2})-(\vec{\epsilon}_{20})}{(\vec{\epsilon}_{2})}$', fontsize = 20)

ax4.legend(fontsize = 14, loc = 'upper right')
fig4.savefig(f'{folder}/all_{names}_LS.png' ,bbox_inches = "tight")
#     # Plot the square

plt.figure()

plt.hist(rel_err1_errxy_2IP, bins = 'auto')
plt.title(r'Component-wise histogram of $\Delta{\vec{\epsilon}}/\vec{\epsilon}$')
plt.xlabel('Relative error')
plt.ylabel('Recurrence')
plt.savefig(f'{folder}/histogram_rel_err.png')



# %%
#TREES PLOT FOR 4 DIMENSIONS problems with no error.

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
folder = '../examples/trees/plots/noerrxy12'
dict = {'mu0x':[0.01],'mu0y':[0.01],'px':[0.01],'py':[0.01]}
title = r'Reconstruction of target $\mathbf{\vec{\epsilon}}$ by Luminosity, varying separation & crossing angle ($\mathbf{\vec{\mu}}$,$\mathbf{\vec{\theta}})$'
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
    jobs = jobs

    #square definition


    path = f'../examples/trees/tree_noerrxy12/{jobs+1:03}'
    js = open(path + '/output_data.json')
    data_out = json.load(js)
    frac_int_eps = np.concatenate([frac_int_eps,[[data_out['eps'][ii][jj]/(2.3e-6)-1 for jj in\
                  range(len(data_out['eps'][ii]))] for ii in range(len(data_out['eps']))]],axis = 0)
    frac_init_est_eps = np.concatenate([frac_init_est_eps,[[data_out['est_eps'][ii][jj]/(2.3e-6)-1 for jj in\
                  range(len(data_out['est_eps'][ii]))] for ii in range(len(data_out['est_eps']))]],axis = 0)

    rel_err_noerrxy12 = np.append(rel_err_noerrxy12,[np.linalg.norm([(data_out['eps'][ii][jj]-data_out['est_eps'][ii][jj]) for jj in range(len(data_out['eps'][ii]))])/np.linalg.norm(data_out['eps'][ii]) for ii in range(len(data_out['eps']))])
    rel_err1_noerrxy12 = np.append(rel_err1_noerrxy12,[[(data_out['eps'][ii][jj]-data_out['est_eps'][ii][jj])/data_out['eps'][ii][jj] for jj in range(len(data_out['eps'][ii]))] for ii in range(len(data_out['eps']))])


rel_err_noerrxy12, indeces = reject_outliers(rel_err_noerrxy12)
rel_err1_noerrxy12 = np.array(rel_err1_noerrxy12)[indeces.repeat(4)]
frac_int_eps_x1 = np.array([frac_int_eps[ii][0] for ii in range(len(frac_int_eps))])[indeces]
frac_int_eps_x2 = np.array([frac_int_eps[ii][2] for ii in range(len(frac_int_eps))])[indeces]
frac_int_eps_y1 = np.array([frac_int_eps[ii][1] for ii in range(len(frac_int_eps))])[indeces]
frac_int_eps_y2 = np.array([frac_int_eps[ii][3] for ii in range(len(frac_int_eps))])[indeces]
frac_init_est_eps_x1 = np.array([frac_init_est_eps[ii][0] for ii in range(len(frac_init_est_eps))])[indeces]
frac_init_est_eps_x2 = np.array([frac_init_est_eps[ii][2] for ii in range(len(frac_init_est_eps))])[indeces]
frac_init_est_eps_y1 = np.array([frac_init_est_eps[ii][1] for ii in range(len(frac_init_est_eps))])[indeces]
frac_init_est_eps_y2 = np.array([frac_init_est_eps[ii][3] for ii in range(len(frac_init_est_eps))])[indeces]


fig,ax = plt.subplots(1,2,figsize = (20,10))
fig.suptitle(title, fontsize=20)
x = [-0.2, 0.2, 0.2, -0.2, -0.2]
y = [-0.2, -0.2, 0.2, 0.2, -0.2]
ax[0].plot(x, y, linewidth = 4)
ax[1].plot(x, y, linewidth = 4)
ax[0].scatter(frac_int_eps_x1[0],frac_int_eps_x2[0],marker   = 'x', c = 'black',s = 100,linewidths=3, label = r'target $\mathbf{\vec{\epsilon}_{x}}$')
    
ax[1].scatter(frac_int_eps_y1[0],frac_int_eps_y2[0],marker   = 'x', c = 'black',s = 100,
linewidths=3, label = r'target $\mathbf{\vec{\epsilon}_{y}})$')
ax[0].scatter(frac_init_est_eps_x1[0],frac_init_est_eps_x2[0], c = 'orange', s = 30,label = r'estimated $\mathbf{\vec{\epsilon}_{x}}$')
ax[1].scatter(frac_init_est_eps_y1[0],frac_init_est_eps_y2[0], c = 'orange', s = 30,label = r'estimated $\mathbf{\vec{\epsilon}_{y}}$')
ax[0].scatter(0,0, c = 'red',s = 40, label = r'intial guess ($\mathbf{\vec{\epsilon}_{x0}}$)')
ax[1].scatter(0,0, c = 'red',s = 40, label = r'initial guess ($\mathbf{\vec{\epsilon}_{y0}}$')
ax[0].set_xlabel(r'$\frac{(\vec{\epsilon}_{x})_{1}-(\vec{\epsilon}_{x0})_{1}}{(\vec{\epsilon}_{x})_{1}}$', fontsize = 20)
ax[0].set_ylabel(r'$\frac{(\vec{\epsilon}_{x})_{2}-(\vec{\epsilon}_{x0})_{2}}{(\vec{\epsilon}_{x})_{2}}$', fontsize = 20)
ax[1].set_xlabel(r'$\frac{(\vec{\epsilon}_{y})_{1}-(\vec{\epsilon}_{y0})_{1}}{(\vec{\epsilon}_{y})_{1}}$', fontsize = 20)
ax[1].set_ylabel(r'$\frac{(\vec{\epsilon}_{y})_{2}-(\vec{\epsilon}_{y0})_{2}}{(\vec{\epsilon}_{y})_{2}}$', fontsize = 20)

ax[0].legend(fontsize = 20, loc = 'upper right')
ax[1].legend(fontsize = 20, loc = 'upper right')
ax[0].set_xlim((-0.25,0.25))
ax[0].set_ylim((-0.25,0.25))
ax[1].set_xlim((-0.25,0.25))
ax[1].set_ylim((-0.25,0.25))
fig.savefig(f'{folder}/one_eps/{names}_LS.png')

# ########## COMPUTATION ITERATIONS

random.seed(42)
epsx1 = random.uniform(2.3e-6-4.6e-7,2.3e-6+4.6e-7)
epsy1 = random.uniform(2.3e-6-4.6e-7,2.3e-6+4.6e-7)
epsx2 = random.uniform(2.3e-6-4.6e-7,2.3e-6+4.6e-7)
epsy2 = random.uniform(2.3e-6-4.6e-7,2.3e-6+4.6e-7)
eps = [epsx1,epsy1,epsx2,epsy2]


dict_shift = dict
with open(f"output_zoom.txt","w") as h,contextlib.redirect_stdout(h):
    [est_eps,function, jacobian, time_LS, iterations] = inv_g_xy12.inv_gauss_single_par(epsx1,epsy1,epsx2,epsy2,dict_shift, verbose = 2, iteration = 0)




with open('output_zoom.txt') as f:
    contents = f.read()
from scipy.stats import gaussian_kde
fig1,ax1 = plt.subplots(1,2,figsize = (20,10))
it_norm_x1 = [(eval(ii.split('\nObjective')[0])[0]/(2.3e-6) -1) for ii in contents.split('=============\nSolution: ')[1:]]
it_norm_x2 = [(eval(ii.split('\nObjective')[0])[2]/(2.3e-6) -1) for ii in contents.split('=============\nSolution: ')[1:]]
it_norm_y1 = [(eval(ii.split('\nObjective')[0])[1]/(2.3e-6) -1) for ii in contents.split('=============\nSolution: ')[1:]]
it_norm_y2 = [(eval(ii.split('\nObjective')[0])[3]/(2.3e-6) -1) for ii in contents.split('=============\nSolution: ')[1:]]

cmap = plt.cm.get_cmap('viridis')


scatter0 = ax1[0].scatter(it_norm_x1[1:],it_norm_x2[1:],c=np.arange(len(it_norm_x1)-1),cmap = cmap)
scatter1 = ax1[1].scatter(it_norm_y1[1:],it_norm_y2[1:],c=np.arange(len(it_norm_y1)-1), cmap = cmap)    
ax1[0].scatter(0,0, c = 'red',s = 40)
ax1[1].scatter(0,0, c = 'red',s = 40)
ax1[0].scatter(frac_int_eps_x1[0],frac_int_eps_x2[0],marker   = 'x', c = 'black',s = 100,linewidths=3,zorder = 3)
ax1[1].scatter(frac_int_eps_y1[0],frac_int_eps_y2[0],marker   = 'x', c = 'black',s = 100,
linewidths=3)
fig1.suptitle(titles_iter, fontsize=20)
ax1[0].set_xlabel(r'$\frac{(\vec{\epsilon}_{x})_{1}-(\vec{\epsilon}_{x0})_{1}}{(\vec{\epsilon}_{x})_{1}}$', fontsize = 20)
ax1[0].set_ylabel(r'$\frac{(\vec{\epsilon}_{x})_{2}-(\vec{\epsilon}_{x0})_{2}}{(\vec{\epsilon}_{x})_{2}}$', fontsize = 20)
ax1[1].set_xlabel(r'$\frac{(\vec{\epsilon}_{y})_{1}-(\vec{\epsilon}_{y0})_{1}}{(\vec{\epsilon}_{y})_{1}}$', fontsize = 20)
ax1[1].set_ylabel(r'$\frac{(\vec{\epsilon}_{y})_{2}-(\vec{\epsilon}_{y0})_{2}}{(\vec{\epsilon}_{y})_{2}}$', fontsize = 20)
ax1[0].set_xlim((-0.25,0.25))
ax1[0].set_ylim((-0.25,0.25))
ax1[1].set_xlim((-0.25,0.25))
ax1[1].set_ylim((-0.25,0.25))
cbar1 = fig1.colorbar(scatter0, ax=ax1[0])
cbar2 = fig1.colorbar(scatter1, ax=ax1[1])
fig1.savefig(f'{folder}/one_eps/{names}_iter_LS.png')

# ########################

fig2,ax2 = plt.subplots(1,2,figsize = (20,10))
ax2[0].plot(it_norm_x1,it_norm_x2, zorder=1)
ax2[1].plot(it_norm_y1,it_norm_y2, zorder=1)
scatter0 = ax2[0].scatter(it_norm_x1[1:],it_norm_x2[1:],c=np.arange(len(it_norm_x1)-1),cmap = cmap, zorder=2)
scatter1 = ax2[1].scatter(it_norm_y1[1:],it_norm_y2[1:],c=np.arange(len(it_norm_y1)-1), cmap = cmap, zorder=2)    
ax2[0].scatter(0,0, c = 'red',s = 40)
ax2[1].scatter(0,0, c = 'red',s = 40)
ax2[0].scatter(frac_int_eps_x1[0],frac_int_eps_x2[0],marker   = 'x', c = 'black',s = 100,linewidths=3,zorder = 3)
ax2[1].scatter(frac_int_eps_y1[0],frac_int_eps_y2[0],marker   = 'x', c = 'black',s = 100,linewidths=3,zorder = 3)

fig2.suptitle(titles_iter_zoom, fontsize=20)
ax2[0].set_xlabel(r'$\frac{(\vec{\epsilon}_{x})_{1}-(\vec{\epsilon}_{x0})_{1}}{(\vec{\epsilon}_{x})_{1}}$', fontsize = 20)
ax2[0].set_ylabel(r'$\frac{(\vec{\epsilon}_{x})_{2}-(\vec{\epsilon}_{x0})_{2}}{(\vec{\epsilon}_{x})_{2}}$', fontsize = 20)
ax2[1].set_xlabel(r'$\frac{(\vec{\epsilon}_{y})_{1}-(\vec{\epsilon}_{y0})_{1}}{(\vec{\epsilon}_{y})_{1}}$', fontsize = 20)
ax2[1].set_ylabel(r'$\frac{(\vec{\epsilon}_{y})_{2}-(\vec{\epsilon}_{y0})_{2}}{(\vec{\epsilon}_{y})_{2}}$', fontsize = 20)


cbar1 = fig2.colorbar(scatter0, ax=ax2[0])
cbar2 = fig2.colorbar(scatter1, ax=ax2[1])
fig2.savefig(f'{folder}/one_eps/zoom_{names}_iter_LS.png')
# ################

iter_x1 = [eval(ii.split('\nObjective')[0])[0] for ii in contents.split('=============\nSolution: ')[1:]]
iter_x2 = [eval(ii.split('\nObjective')[0])[2] for ii in contents.split('=============\nSolution: ')[1:]]
iter_y1 = [eval(ii.split('\nObjective')[0])[1] for ii in contents.split('=============\nSolution: ')[1:]]
iter_y2 = [eval(ii.split('\nObjective')[0])[3] for ii in contents.split('=============\nSolution: ')[1:]]

L_target = lumi.L(epsx1,epsy1,epsx2,epsy2, *parameters_sym) 
L_iteration = [lumi.L(iter_x1[j],iter_y1[j],iter_x2[j],iter_y2[j], *parameters_sym) for j in range(len(iter_x1))]

rel_err_L = [abs(L_target-L_iteration[ii+len(L_iteration)-100])/L_target for ii in range(len(L_iteration)-(len(L_iteration)-100))]

fig3,ax3 = plt.subplots(figsize = (20,10))
fig3.suptitle('Relative error on Luminosity estimation over the last 100 iterations', fontsize=20)
ax3.plot(np.linspace(len(L_iteration)-100,len(L_iteration)-1,len(L_iteration)-(len(L_iteration)-100)),rel_err_L)
ax3.set_xlabel('Iterations of Least Squares', fontsize = 20)
ax3.set_ylabel(r'$\frac{(\mathcal{L}_{target}-\mathcal{L}_{iter})}{\mathcal{L}_{target}}$', fontsize = 20)
fig3.savefig(f'{folder}/one_eps/rel_err_L_{names}.png')

# #################

fig4,ax4 = plt.subplots(1,2,figsize = (20,10))
fig4.suptitle(title, fontsize=20)


# Plot the square
ax4[0].plot(x, y, linewidth = 4)
ax4[1].plot(x, y, linewidth = 4)

ax4[0].scatter(frac_int_eps_x1,frac_int_eps_x2,marker   = 'x', c = 'black',s = 100,linewidths=3, label = r'target $\mathbf{\vec{\epsilon}_{x}}$')
ax4[1].scatter(frac_int_eps_y1,frac_int_eps_y2,marker   = 'x', c = 'black',s = 100,
linewidths=3, label = r'target $\mathbf{\vec{\epsilon}_{y}})$')
ax4[0].scatter(frac_init_est_eps_x1,frac_init_est_eps_x2, c = 'orange', s = 10,label = r'estimated $\mathbf{\vec{\epsilon}_{x}}$')
ax4[1].scatter(frac_init_est_eps_y1,frac_init_est_eps_y2, c = 'orange', s = 10,label = r'estimated $\mathbf{\vec{\epsilon}_{y}}$')
ax4[0].scatter(0,0, c = 'red',s = 40, label = r'intial guess ($\mathbf{\vec{\epsilon}_{x0}}$)')
ax4[1].scatter(0,0, c = 'red',s = 40, label = r'initial guess ($\mathbf{\vec{\epsilon}_{y0}}$')
ax4[0].set_xlabel(r'$\frac{(\vec{\epsilon}_{x})_{1}-(\vec{\epsilon}_{x0})_{1}}{(\vec{\epsilon}_{x})_{1}}$', fontsize = 20)
ax4[0].set_ylabel(r'$\frac{(\vec{\epsilon}_{x})_{2}-(\vec{\epsilon}_{x0})_{2}}{(\vec{\epsilon}_{x})_{2}}$', fontsize = 20)
ax4[1].set_xlabel(r'$\frac{(\vec{\epsilon}_{y})_{1}-(\vec{\epsilon}_{y0})_{1}}{(\vec{\epsilon}_{y})_{1}}$', fontsize = 20)
ax4[1].set_ylabel(r'$\frac{(\vec{\epsilon}_{y})_{2}-(\vec{\epsilon}_{y0})_{2}}{(\vec{\epsilon}_{y})_{2}}$', fontsize = 20)

ax4[0].legend(fontsize = 20, loc = 'upper right')
ax4[1].legend(fontsize = 20, loc = 'upper right')
fig4.savefig(f'{folder}/all_{names}_LS.png')

plt.figure()

plt.hist(rel_err1_noerrxy12, bins = 'auto')
plt.title(r'Component-wise histogram of $\Delta{\vec{\epsilon}}/\vec{\epsilon}$')
plt.xlabel('Relative error')
plt.ylabel('Recurrence')
plt.savefig(f'{folder}/histogram_rel_err.png')

# %%

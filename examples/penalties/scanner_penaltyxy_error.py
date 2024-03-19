#PLOT OF THE PENALTY FUNCTION ON EPSXY, WITH ERROR
#%%
import contextlib
import random
import numpy as np
from numpy import linalg as LA
from inversion_fun import Inv_gauss_xy as inv_g
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.colors as mcolors



[f,nb,N,energy_tot] = [11245,2736,1.4e11,6800]
[dif_mu0x,dif_mu0y] = [0,0]
[dif_px,dif_py] = [320e-6,0]#[160e-6,160e-6]
[alphax,alphay] = [0,0]
[sigmaz] = [0.35]
[betax,betay] =[0.3,0.3]
[deltap_p0] = [0]
[dmu0x,dmu0y] = [0,0]
[dpx,dpy] = [0,0]

parameters_sym_x = [f,nb,N,N,energy_tot,energy_tot,dif_mu0x,dif_mu0y,dif_py,dif_px, alphax,alphay,alphax,alphay,sigmaz,sigmaz,
            betax,betay,betax,betay,deltap_p0,deltap_p0,dmu0x,dmu0y,-dmu0x,-dmu0y,dpx,dpy,-dpx,-dpy]

parameters_sym_y = [f,nb,N,N,energy_tot,energy_tot,dif_mu0x,dif_mu0y,dif_px,dif_py, alphax,alphay,alphax,alphay,sigmaz,sigmaz,
            betax,betay,betax,betay,deltap_p0,deltap_p0,dmu0x,dmu0y,-dmu0x,-dmu0y,dpx,dpy,-dpx,-dpy]


dict_shift = {'mu0x':[0.01]}
random.seed(41)

LS = []
eps = []
sol_LS = np.zeros([2+1,2])
f_sol = np.zeros([2+1,4])
Jac_sol = np.zeros([2+1,4,2])
d = {}
for i in range(1):
    eps1 = random.uniform(0.8*2.3e-6,1.2*2.3e-6)
    eps2 = random.uniform(0.8*2.3e-6,1.2*2.3e-6)
    eps = [eps1,eps2]
    with open(f"output_zoom.txt","w") as h,contextlib.redirect_stdout(h):
        [sol_LS[i+1],f_sol[i+1],Jac_sol[i+1],timing_LS,nfev_LS] = inv_g.inv_gauss_xy_randerr_2(eps1,eps2,dict_shift,verbose=2, para = [parameters_sym_x,parameters_sym_y])
    LS.append([timing_LS,nfev_LS])
rel_err = {'est_eps': sol_LS, 'eps': eps, 'f_sol': f_sol, 'Jacobian': Jac_sol, 'LS': LS}




# %%
def compute_points_on_line(x1, y1, x2, y2, num_points, y_min=None, y_max=None):
    # Calculate the slope (m)
    m = (y2 - y1) / (x2 - x1)
    
    # Calculate the y-intercept (c)
    c = y1 - m * x1
    
    # Determine the range of y-values
    y_min = min(y1, y2) if y_min is None else min(min(y1, y2), y_min)
    y_max = max(y1, y2) if y_max is None else max(max(y1, y2), y_max)
    
    # Compute the corresponding x-values for each y-value within the range
    points = []
    for i in range(num_points):
        y = y_min + (y_max - y_min) * i / (num_points - 1)
        x = (y - c) / m
        points.append([x, y])
    
    return points

def points_along_line(start_point, direction_vector, num_points):
    """
    Generate points along a line given a starting point and a direction vector.
    :param start_point: The starting point of the line as a NumPy array [x0, y0].
    :param direction_vector: The direction vector of the line as a NumPy array [dx, dy].
    :param num_points: The number of points to generate along the line.
    :return: A list of NumPy arrays representing the points along the line.
    """
    t_values = np.linspace(0, 1, num_points)
    points = [start_point + t * direction_vector for t in t_values]
    return points


# %%
import ast
import math

with open('output_zoom_epsxy_err.txt') as f:
    contents = f.read()

constants = list(eval(contents.split('=============\nSolution: ')[0].split('\n')[-2]))
paramx = ast.literal_eval(contents.split('[0.01]\n')[2].split('\n')[0])
paramy = ast.literal_eval(contents.split('[0.01]\n')[2].split('\n')[1])


def L_over_eps_xy(epsx,epsy, par):
    return inv_g.L_over_parameters_sym(epsx,epsy,epsx,epsy, par)

def penalty_square(xy):
    penalty = (L_over_eps_xy(xy[0],xy[1],parameters_sym_x)-constants[0])**2
    penalty+= (L_over_eps_xy(xy[0],xy[1],parameters_sym_y)-constants[1])**2
    keys = list(paramx['parx'].keys())
    ax = ['x','y']
    for i in range(int(len(constants)/2)-1):
       for j in range(2):
            param = [paramx,paramy][j]
            penalty += (L_over_eps_xy(xy[0],xy[1],param[f'par{ax[j]}'][keys[i]])-constants[(i+2)+j])**2
    return [np.sqrt(penalty)]



iter_x = [eval(ii.split('\n')[0])[0] for ii in contents.split('=============\nSolution: ')[1:]]
iter_y = [eval(ii.split('\n')[0])[1] for ii in contents.split('=============\nSolution: ')[1:]]
L_eps_iter = [penalty_square([iter_x[ii+3],iter_y[ii+3]]) for ii in range(len(iter_y)-3)]



vec_epsx = np.linspace(0.8*2.3e-6,1.2*2.3e-6,300)
vec_epsy = np.linspace(0.8*2.3e-6,1.2*2.3e-6,300)

xv,yv = np.array(np.meshgrid(vec_epsx, vec_epsy))
Z = [[penalty_square([xv[j][ii],yv[j][ii]]) for ii in range(len(xv[j]))] for j in range(len(xv))]
L_eps = np.flipud(Z)

x = 2.3e-6
y = 2.3e-6
h = 1e-6
df_dx = (penalty_square([x + h, y])[0] - penalty_square([x - h, y])[0]) / (2 * h)
df_dy = (penalty_square([x, y + h])[0] - penalty_square([x, y - h])[0]) / (2 * h)

gradient = np.array([df_dx,df_dy])
vector_line = np.array([-gradient[1],gradient[0]])/LA.norm(np.array([-gradient[1],gradient[0]]))
c = points_along_line([rel_err['eps'][0],rel_err['eps'][1]],-vector_line*1e-6,100)
d = points_along_line([rel_err['eps'][0],rel_err['eps'][1]],vector_line*1e-6,100)
points_line = np.concatenate([c[::-2],d])

x_line = [rel_err['eps'][0] for ii in points_line]
y_line = [ii[1] for ii in points_line]
L_eps_line = [penalty_square([x_line[ii],y_line[ii]]) for ii in range(len(x_line))]
dist = [LA.norm(np.array(points_line[ii])-np.array(points_line[0])) for ii in range(len(x_line))]

mesh_xy = np.array(np.meshgrid(vec_epsx, vec_epsy)).T.reshape(-1,2)
L_eps_vec = [np.sqrt(np.sum(penalty_square(ii))) for ii in mesh_xy]
mesh_x = [ii[0] for ii in mesh_xy]
mesh_y = [ii[1] for ii in mesh_xy]

cmap = plt.cm.get_cmap('viridis')
cmap2 = plt.cm.get_cmap('plasma')
scatter0 = plt.scatter(mesh_x,mesh_y,c=L_eps_vec,cmap = cmap,norm=mcolors.LogNorm(),zorder = 0)#vmin=min(L_eps_vec), vmax=max(L_eps_vec)),zorder = 0)
plt.scatter(rel_err['eps'][0],rel_err['eps'][1],marker = 'x', c = 'black',zorder = 3.5,linewidths=3, s = 100,label = 'real emittance')
scatter1= plt.scatter(iter_x,iter_y,zorder = 2,c=np.arange(len(iter_x)),cmap = cmap2)
plt.legend(fontsize = 20, loc = 'upper center')
plt.xlim(0.8*2.3e-6,1.2*2.3e-6)
plt.ylim(0.8*2.3e-6,1.2*2.3e-6)
scatter1= plt.scatter(iter_x,iter_y,zorder = 2,c=np.arange(len(iter_x)),cmap = cmap2)
plt.scatter(iter_x[0],iter_y[0],zorder = 3,c = 'red')

squarex = [0.98*rel_err['eps'][0],1.02*rel_err['eps'][0],1.02*rel_err['eps'][0],0.98*rel_err['eps'][0],0.98*rel_err['eps'][0]]
squarey = [0.98*rel_err['eps'][1],0.98*rel_err['eps'][1],1.02*rel_err['eps'][1],1.02*rel_err['eps'][1],0.98*rel_err['eps'][1]]

plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.xlabel('epsilonx', fontsize = 25)
plt.ylabel('epsilony', fontsize = 25)
plt.title('Penality of the LS', fontsize = 30)
plt.gca().yaxis.offsetText.set_fontsize(20)
plt.gca().xaxis.offsetText.set_fontsize(20) 
plt.tight_layout()
cbar1 = plt.colorbar(scatter0)
cbar1.ax.get_yaxis().labelpad = 16
cbar1.set_label( "Penalty values", rotation=270, fontsize = 20)
cbar1.ax.tick_params(labelsize=20) 
cbar = plt.colorbar(scatter1)
cbar.ax.get_yaxis().labelpad = 16
cbar.set_label( "# iteration LS", rotation=270, fontsize = 20)
cbar.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
cbar.ax.tick_params(labelsize=20) 
plt.savefig('plots/penalty_xy_errsum.png',dpi = 200, bbox_inches='tight')



# %%

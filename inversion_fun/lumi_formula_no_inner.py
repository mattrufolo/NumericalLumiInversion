#%%
import numpy as np
from scipy import integrate
# import sys
# sys.path.append("/var/data/mrufolo/Inverting_luminosity")
from inversion_fun import particle
from numba import njit
import time
import random

restEnergyProton_GeV=0.93827231


@njit
def beta(z, beta0, alpha_z0):
    '''Beta function in drift space'''
    return beta0-2*alpha_z0*z+(1+alpha_z0**2)/beta0*z**2

@njit
def dispersion(z, d0, dp0):
    '''Dispersion in drift space'''
    return d0+z*dp0
@njit
def sigma(beta, epsilon0, betagamma):
    '''Betatronic sigma'''
    return np.sqrt(beta*epsilon0/betagamma)
@njit
def sx1(z, epsilonx_1, energy_tot1, alphax_1, betax_1,deltap_p0_1,dmu0x_1,dpx_1):
    '''The sigma_x of B1, quadratic sum of betatronic and dispersive sigma'''
    #energy_tot1, alphax_1, betax_1,deltap_p0_1,dmu0x_1,dpx_1 = par_x1
    relGamma1=energy_tot1/restEnergyProton_GeV
    relBeta1=np.sqrt(1.-relGamma1**-2)
    relBetaGamma1=relGamma1*relBeta1
    return np.sqrt(sigma(beta(z, betax_1, alphax_1), epsilonx_1, relBetaGamma1)**2 \
    + (dispersion(z, relBeta1*dmu0x_1, relBeta1*dpx_1)*deltap_p0_1)**2)
@njit
def sy1(z, epsilony_1, energy_tot1, alphay_1, betay_1,deltap_p0_1,dmu0y_1,dpy_1):
    '''The sigma_y of B1, quadratic sum of betatronic and dispersive sigma'''
    #energy_tot1, alphay_1, betay_1,deltap_p0_1,dmu0y_1,dpy_1 = par_y1
    relGamma1=energy_tot1/restEnergyProton_GeV
    relBeta1=np.sqrt(1.-relGamma1**-2)
    relBetaGamma1=relGamma1*relBeta1
    return np.sqrt(sigma(beta(z, betay_1, alphay_1), epsilony_1, relBetaGamma1)**2 \
    + (dispersion(z, relBeta1*dmu0y_1, relBeta1*dpy_1)*deltap_p0_1)**2)
@njit
def sx2(z, epsilonx_2, energy_tot2, alphax_2, betax_2,deltap_p0_2,dmu0x_2,dpx_2):
    '''The sigma_x of B2, quadratic sum of betatronic and dispersive sigma'''
    #energy_tot2, alphax_2, betax_2,deltap_p0_2,dmu0x_2,dpx_2 = par_x2
    relGamma2=energy_tot2/restEnergyProton_GeV
    relBeta2=np.sqrt(1.-relGamma2**-2)
    relBetaGamma2=relGamma2*relBeta2
    return np.sqrt(sigma(beta(z, betax_2, alphax_2), epsilonx_2, relBetaGamma2)**2 \
    + (dispersion(z, relBeta2*dmu0x_2, relBeta2*dpx_2)*deltap_p0_2)**2)
@njit
def sy2(z, epsilony_2, energy_tot2, alphay_2, betay_2,deltap_p0_2,dmu0y_2,dpy_2):
    '''The sigma_y of B2, quadratic sum of betatronic and dispersive sigma'''
    #energy_tot2, alphay_2, betay_2,deltap_p0_2,dmu0y_2,dpy_2 = par_y2
    relGamma2=energy_tot2/restEnergyProton_GeV
    relBeta2=np.sqrt(1.-relGamma2**-2)
    relBetaGamma2=relGamma2*relBeta2
    return np.sqrt(sigma(beta(z, betay_2, alphay_2), epsilony_2, relBetaGamma2)**2 \
    + (dispersion(z, relBeta2*dmu0y_2, relBeta2*dpy_2)*deltap_p0_2)**2)

@njit
def dif_mx(z, dif_mu0x,dif_px):
    '''The difference of mu_x betweem B1 and B2 as straight line'''
    #mu0x_1,px_1 = mu_p_x1
    return dif_mu0x + dif_px*z
@njit
def dif_my(z, dif_mu0y,dif_py):
    '''The difference of mu_y betweem B1 and B2 as straight line'''
    #mu0x_1,px_1 = mu_p_x1
    return dif_mu0y + dif_py*z

@njit
def kernel_single_integral(z,epsilonx_1,epsilony_1, epsilonx_2, epsilony_2, energy_tot1,energy_tot2,
                        dif_mu0x,dif_mu0y,dif_px,dif_py, alphax_1,alphay_1,alphax_2,alphay_2,sigmaz_1,sigmaz_2,
                        betax_1,betay_1,betax_2,betay_2,deltap_p0_1,deltap_p0_2,dmu0x_1,dmu0y_1,dmu0x_2,dmu0y_2,dpx_1,dpy_1,dpx_2,dpy_2):
    #energy_tot1,energy_tot2,mu0x_1,mu0y_1,mu0x_2,mu0y_2,px_1,py_1,px_2,py_2, alphax_1,alphay_1,alphax_2,alphay_2,sigmaz_1,sigmaz_2,\
    #betax_1,betay_1,betax_2,betay_2,deltap_p0_1,deltap_p0_2,dmu0x_1,dmu0y_1,dmu0x_2,dmu0y_2,dpx_1,dpy_1,dpx_2,dpy_2 = par
    par_x1 = [energy_tot1, alphax_1, betax_1,deltap_p0_1,dmu0x_1,dpx_1] 
    par_y1 = [energy_tot1, alphay_1, betay_1,deltap_p0_1,dmu0y_1,dpy_1] 
    par_x2 = [energy_tot2, alphax_2, betax_2,deltap_p0_2,dmu0x_2,dpx_2] 
    par_y2 = [energy_tot2, alphay_2, betay_2,deltap_p0_2,dmu0y_2,dpy_2] 
    relGamma1=energy_tot1/restEnergyProton_GeV
    relBeta1=np.sqrt(1.-relGamma1**-2)
    relGamma2=energy_tot2/restEnergyProton_GeV
    relBeta2=np.sqrt(1.-relGamma2**-2)
    # return np.exp(0.5*(-(mx1(z,*mu_p_x1) - mx2(z,*mu_p_x2))**2/(sx1(z,epsilonx_1,*par_x1)**2 + sx2(z,epsilonx_2,*par_x2)**2) \
    # -(my1(z,*mu_p_y1) - my2(z,*mu_p_y2))**2/(sy1(z,epsilony_1,*par_y1)**2 + sy2(z,epsilony_2,*par_y2)**2) \
    # -((relBeta1+relBeta2)**2*z**2)/(relBeta2**2*sigmaz_1**2 + relBeta1**2*sigmaz_2**2))) \
    # /np.sqrt((sx1(z,epsilonx_1,*par_x1)**2 + sx2(z,epsilonx_2,*par_x2)**2)*(sy1(z,epsilony_1,*par_y1)**2 + sy2(z,epsilony_2,*par_y2)**2)*(sigmaz_1**2 + sigmaz_2**2))
    return np.exp(0.5*(-(dif_mx(z, dif_mu0x,dif_px))**2/(sx1(z,epsilonx_1,energy_tot1, alphax_1, betax_1,deltap_p0_1,dmu0x_1,dpx_1)**2\
     + sx2(z,epsilonx_2,energy_tot2, alphax_2, betax_2,deltap_p0_2,dmu0x_2,dpx_2)**2) -(dif_my(z, dif_mu0y,dif_py))**2\
        /(sy1(z,epsilony_1,energy_tot1, alphay_1, betay_1,deltap_p0_1,dmu0y_1,dpy_1)**2 + sy2(z,epsilony_2,energy_tot2, alphay_2, betay_2,deltap_p0_2,dmu0y_2,dpy_2)**2)-\
        ((relBeta1+relBeta2)**2*z**2)/(relBeta2**2*sigmaz_1**2 + relBeta1**2*sigmaz_2**2))) \
    /np.sqrt((sx1(z,epsilonx_1,energy_tot1, alphax_1, betax_1,deltap_p0_1,dmu0x_1,dpx_1)**2 + sx2(z,epsilonx_2,energy_tot2, alphax_2, betax_2,deltap_p0_2,dmu0x_2,dpx_2)**2)\
    *(sy1(z,epsilony_1,energy_tot1, alphay_1, betay_1,deltap_p0_1,dmu0y_1,dpy_1)**2 + sy2(z,epsilony_2,energy_tot2, alphay_2, betay_2,deltap_p0_2,dmu0y_2,dpy_2)**2)*(sigmaz_1**2 + sigmaz_2**2))

def L(epsilonx_1, epsilony_1,
      epsilonx_2, epsilony_2,
      *parameters):
    '''
    Returns luminosity in Hz/cm^2.

    f: revolution frequency
    nb: number of colliding bunch per beam in the specific Interaction Point (IP).
    N1,N2: B1,2 number of particle per bunch
    mu0x1,mu0x2,mu0y1,mu0uy2: horizontal/vertical position at the IP of B1,2[m]
    px1,px2,py1,py2: px,py at the IP of B1,2,
    energy_tot1,emergu_tot2: total energy of the B1,2 [GeV]
    deltap_p0_1,delta_p0_2: rms momentum spread of B1,2 (formulas assume Gaussian off-momentum distribution)
    epsilon_x1,epsilon_x2, epsilon_y1,epsilon_y2: horizontal/vertical normalized emittances of B1,2 [m rad]
    sigma_z1,sigma_z2: rms longitudinal spread in z of B1,2 [m]
    beta_x1,beta_x2,beta_y1,beta_y2: horizontal/vertical beta-function at IP of B1,2 [m]
    alpha_x1,alpha_x2,alpha_y1,alpha_y2: horizontal/vertical alpha-function at IP of B1,2 
    dx_1,dx_2,dy_1,dy_2: horizontal/vertical dispersion-function at IP of B1,2 [m]
    dpx_1,dpx_2,dpy_1,dpy_2 horizontal/vertical differential-dispersion-function IP of B1,2

    '''
    #print(len(parameters))
    

    f,nb,N1,N2,energy_tot1,energy_tot2,dif_mu0x,dif_mu0y,dif_px,dif_py, alphax_1,alphay_1,alphax_2,alphay_2,sigmaz_1,sigmaz_2,\
    betax_1,betay_1,betax_2,betay_2,deltap_p0_1,deltap_p0_2,dmu0x_1,dmu0y_1,dmu0x_2,dmu0y_2,dpx_1,dpy_1,dpx_2,dpy_2 = parameters
    # f,nb,N1,energy_tot1,dif_mu0x,dif_mu0y,dif_px,dif_py, alphax_1,alphay_1,alphax_2,alphay_2,sigmaz_1,sigmaz_2,\
    # betax_1,betay_1,betax_2,betay_2,deltap_p0_1,deltap_p0_2,dmu0x_1,dmu0y_1,dmu0x_2,dmu0y_2,dpx_1,dpy_1,dpx_2,dpy_2 = parameters
    N2 = N1
    energy_tot2 = energy_tot1
    par = [energy_tot1,energy_tot2,dif_mu0x,dif_mu0y,dif_px,dif_py, alphax_1,alphay_1,alphax_2,alphay_2,sigmaz_1,sigmaz_2,\
    betax_1,betay_1,betax_2,betay_2,deltap_p0_1,deltap_p0_2,dmu0x_1,dmu0y_1,dmu0x_2,dmu0y_2,dpx_1,dpy_1,dpx_2,dpy_2]
    
    
    particle_1=particle.setTotalEnergy_GeV(energy_tot1)
    relBeta1=particle_1['relativisticBeta']

    particle_2=particle.setTotalEnergy_GeV(energy_tot2)
    relBeta2=particle_2['relativisticBeta']

    c=particle.speedOfLight_m_s

    # # module of B1 speed
    # v0_1=relBeta1*c 
    # # paraxial hypothesis 
    # vx_1=float(v0_1*px_1)
    # vy_1=float(v0_1*py_1)
    # vz_1=float(v0_1*np.sqrt(1-px_1**2-py_1**2))
    # v_1=np.array([vx_1, vy_1, vz_1])

    # v0_2=relBeta2*c # module of B2 speed
    # # Assuming counter rotating B2 ('-' sign)
    # vx_2=float(-v0_2*px_2)
    # vy_2=float(-v0_2*py_2)
    # # assuming px_2**2+py_2**2 < 1
    # vz_2=float(-v0_2*np.sqrt(1-px_2**2-py_2**2))
    # v_2=np.array([vx_2, vy_2, vz_2])

    # diff_v = v_1-v_2
    # cross_v= np.cross(v_1, v_2)

    # normalized to get 1 for the ideal case 
    # NB we assume px_1 and py_1 constant along the z-slices 
    # NOT TRUE FOR CC! In any case the Moeller efficiency is almost equal to 1 in most cases...
    
    ### IN OUR CASE THE DIFFERENCE BETWEEN THE MOELLER EFFICIENCY AND 1 IS IN THE ORDER OF 1e-8
    ### WE DECIDE TO APPROXIMATE THE MOELLER EFFICIENCY IN ORDER TO DEFINE EVERYTHING FROM THE DIFFERENCE OF SEPARATION AND CROSSING
    #Moeller_efficiency=np.sqrt(c**2*np.dot(diff_v,diff_v)-np.dot(cross_v,cross_v))/c**2/2
    Moeller_efficiency = 1
    #print(f'factor: {Moeller_factor}')
    # print(f'efficiency: {Moeller_efficiency}')
    #print(f'diff: {Moeller_efficiency-Moeller_efficiency2}')
    # print(f' far 1:{1-Moeller_efficiency}')
    sigma_z=np.max([sigmaz_1,sigmaz_2])
    start = time.time()
    integral=integrate.quad(lambda z: kernel_single_integral(z,epsilonx_1,epsilony_1, epsilonx_2, epsilony_2, *par), -5*sigma_z, 5*sigma_z)
    end = time.time()
    #print(f'quad: {end-start}')
    #start = time.time()
    #z = np.linspace(-5*sigmaz,5*sigmaz, num = 100)
    #y = [kernel_single_integral(ii) for ii in z] 
    
    #integral2=np.trapz(y,x = z, dx = 1)
    #end = time.time()
    #print(f' trapz:{end-start}')
    L0=f*N1*N2*nb/np.sqrt(2)/np.pi**(3/2)*integral[0]
    #L02=f*N1*N2*nb/np.sqrt(2)/np.pi**(3/2)*integral2
    result= L0*Moeller_efficiency/1e4#[L0*Moeller_efficiency/1e4,L02*Moeller_efficiency/1e4]
    # #print(f'Moeller efficiency: {Moeller_efficiency}')
    #if integral[0]!=0:
    # print(f'Integral Relative Error: {integral[1]/integral[0]}')
    # print(f'==> Luminosity [Hz/cm^2]: {result}')
    return result

# [f,nb,N,energy_tot] = [11245,2736,1.4e11,6800]
# [mu0x,mu0y] = [0,0]
# [px,py] = [160e-6,160e-6]
# [alphax,alphay] = [0,0]
# [sigmaz] = [0.35]
# [betax,betay] =[0.3,0.3]
# [deltap_p0] = [0]
# [dmu0x,dmu0y] = [0,0]
# [dpx,dpy] = [0,0]
# parameters_sym = [f,nb,N,N,energy_tot,energy_tot,mu0x,mu0y,-mu0x,-mu0y,px,py,-px,-py, alphax,alphay,alphax,alphay,sigmaz,sigmaz,
#             betax,betay,betax+1e-4,betay+1e-4,deltap_p0,deltap_p0,dmu0x,dmu0y,-dmu0x,-dmu0y,dpx,dpy,-dpx,-dpy]
# start1 = time.time()
# for i in range(1):
#     random.seed(1)
#     epsx1 = random.uniform(2e-6,2.6e-6)
#     epsx2 = random.uniform(2e-6,2.6e-6)
#     epsy1 = random.uniform(2e-6,2.6e-6)
#     epsy2 = random.uniform(2e-6,2.6e-6)
#     L(epsx1,epsy1,epsx2,epsy2,*parameters_sym)
# end1 = time.time()
# print(end1-start1)
# %%

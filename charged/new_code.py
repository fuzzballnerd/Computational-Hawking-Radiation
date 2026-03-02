# import numpy as np
# import matplotlib.pyplot as plt
# import scipy.interpolate as scp
# import scipy.integrate as sci
# from scipy.integrate import odeint

# mD = 10**(10)
# eD = 3*10**(-4)
# M_pl = 1.22*10**(19)
# g_to_GeV = 5.61*10**(23)
# GeV_to_sec = 6.5821e-25    
# sec_to_yr  = 1.0 / (365.25*24*3600)

# alpha_rough = np.genfromtxt("/home/sarthaka/Downloads/blackhawk_v2.3/scripts/greybody_scripts/greybody_factors/charged/alpha_charged.txt")

# def inter_func(file):
#     some_variable = scp.interp1d(file[:,0], file[:,1], kind='cubic', bounds_error=False, fill_value=0.0)
#     return some_variable

# def alpha_smooth(M, Q):
#         q = abs(Q/M)
#         if q > 0.995:
#             val = 0.0
#         else:
#             alpha_inter = inter_func(alpha_rough)
#             val = alpha_inter(q)
#         return val


# ### for the visualization of alpha_smooth
# ### start
# # M = 9.6*10**(12)*g_to_GeV
# # Q = 0.01*M/M_pl
# # x_data = np.linspace(0,0.9999,10000)
# # y_data = [alpha_smooth(M, q*M) for q in x_data]
# # fig = plt.figure(figsize=(8, 6))
# # plt.plot(x_data, y_data, 'b-')
# # plt.show()
# ### end

# def rplus(M, Q):
#     q = abs(Q/M)
#     return M*(1+np.sqrt(1-(M_pl*q)**2))/M_pl**2

# def de_system(y, t):
#     M, Q = y
#     q = abs(Q/M)
#     if q >= 1.0:
#         q = 1.0
#         dQdt = 0.0
#     if abs(Q) < 10**(-100000): 
#         dQdt = 0.0
#     else:
#         arg = np.pi*rplus(M,Q)**2*mD**2/(eD*abs(Q)) 
#         if arg > 100000000000:
#             dQdt = 0.0
#         else:
#             term = eD**4*abs(Q)**3/(2*mD**2*np.pi**3*rplus(M, Q)**3)
#             dQdt = -term*np.exp(-arg)

#     alpha = alpha_smooth(M, Q)
#     alpha_term = -alpha*M_pl**4/M**2    
#     another_term = Q*dQdt/rplus(M,Q)
#     dMdt = alpha_term + another_term
#     return [dMdt, dQdt]
    
# mass = 9.6*10**(12)*g_to_GeV
# real_q = 0.01*mass/M_pl
# y0 = [mass, real_q]

# t_yr = np.geomspace(1e-1, 10**(12), 10000)
# t_sec = t_yr / sec_to_yr
# t_gev = t_sec / GeV_to_sec

# sol = odeint(de_system, y0, t_gev, rtol=1e-12, atol=1e-12)
# mass_evol = sol[:,0]
# charge_evol = sol[:,1]

# M_g = mass_evol / g_to_GeV

# # def Temp(M, Q):
# #     if Q >= M/M_pl: return 0.0
# #     kappa = np.sqrt(M**2 - (Q*M_pl)**2) / (2*M*(M + np.sqrt(M**2 - (Q*M_pl)**2)) - (Q*M_pl)**2)
# #     return kappa / (2 * np.pi)

# plt.figure(figsize=(8, 6))
# plt.loglog(t_yr, M_g, 'b-', label=r'RN+BSM ($M_{PBH}$)')
# # plt.loglog(t_yr, Temp(mass_evol, charge_evol), 'b-', label=r'RN+BSM ($M_{PBH}$)')

# q_star = charge_evol * M_pl / mass_evol
# q_star = np.clip(q_star, 0, 0.999999999)
# mod_y = -np.log10(1-q_star)

# ax1 = plt.gca()
# ax2 = ax1.twinx()
# ax2.semilogx(t_yr, mod_y, 'r-', label=r'Charge $-Log(1-Q/M)$')

# ax1.set_xlabel("t [yr]")
# ax1.set_ylabel(r"$M_{PBH}$ [g]", color='b')
# ax2.set_ylabel(r"$-Log_{10}[1 - Q^*_D]$", color='r')

# ax1.axvline(380000, color='grey', linestyle='-', linewidth=0.5, alpha=0.5) 
# ax1.axvline(1.38e10, color='orange', linestyle='-', linewidth=1.5, label=r'$t_{universe}$')

# ax1.set_ylim(1e7, 2e13)
# ax1.set_xlim(1e-1, 2e14) 

# plt.title("graph")
# plt.show()




# import numpy as np
# import matplotlib.pyplot as plt
# import scipy.interpolate as scp
# import scipy.integrate as sci
# from scipy.integrate import solve_ivp
# import pylab

# mD = 10**(10)
# eD = 3*10**(-4)
# M_pl = 1.22*10**(19)
# g_to_GeV = 5.61*10**(23)
# GeV_to_sec = 6.5821e-25    
# sec_to_yr  = 1.0 / (365.25*24*3600)

# alpha_rough = np.genfromtxt("/home/sarthaka/Downloads/blackhawk_v2.3/scripts/greybody_scripts/greybody_factors/charged/alpha_charged.txt")

# def inter_func(file):
#     some_variable = scp.interp1d(file[:,0], file[:,1], kind='cubic', bounds_error=False, fill_value=0.0)
#     return some_variable

# def alpha_smooth(M, Q):
#         q = M_pl*abs(Q/M)
#         alpha_inter = inter_func(alpha_rough)
#         val = alpha_inter(q)
#         return val


# ### for the visualization of alpha_smooth
# ### start
# # M = 9.6*10**(12)*g_to_GeV
# # Q = 0.01*M/M_pl
# # x_data = np.linspace(0,0.9999,10000)
# # y_data = [alpha_smooth(M, q*M) for q in x_data]
# # fig = plt.figure(figsize=(8, 6))
# # plt.plot(x_data, y_data, 'b-')
# # plt.show()
# ### end

# def rplus(M, Q):
#     q = abs(Q/M)
#     if M_pl*q >= 0.9:
#          M_pl*q == 0.9
#     return M*(1+np.sqrt(1-(M_pl*q)**2))/M_pl**2


# def de_system(t_sec, y):
#     M, Q = y
#     q = M_pl*abs(Q/M)
#     if q >= 0.85:
#         dQdt_gev = 10**(-28.135)
#         alpha = 10**(-28.135)
#     else:
#         term = eD**4*abs(Q)**3/(2*mD**2*np.pi**3*rplus(M, Q)**3)
#         dQdt_gev = -10**(-100)*term*np.exp(-np.pi*rplus(M,Q)**2*mD**2/(eD*abs(Q)))
#         alpha = alpha_smooth(M, Q)
         
#     alpha_term = -alpha*M_pl**4/M**2
#     another_term = Q*dQdt_gev/rplus(M,Q)
#     dMdt_gev = alpha_term + another_term
#     dQdt_sec = dQdt_gev / GeV_to_sec
#     dMdt_sec = dMdt_gev / GeV_to_sec
#     return [dMdt_sec, dQdt_sec]

# mass = 9.6*10**(12)*g_to_GeV
# real_q = 0.01*mass/M_pl
# y0 = [mass, real_q]

# t_min_yr = 1e-8 
# t_max_yr = 10**(12)
# t_eval_yr = np.geomspace(t_min_yr, t_max_yr, 1000)
# t_eval_sec = t_eval_yr / sec_to_yr

# sol = solve_ivp(de_system, [t_eval_sec[0], t_eval_sec[-1]], y0, 
#                 method='Radau', t_eval=t_eval_sec, rtol=1e-6, atol=1e-10)


# t_yr = sol.t * sec_to_yr
# mass_evol = sol.y[0]
# charge_evol = sol.y[1]

# q_star = charge_evol * M_pl / (mass_evol+10**(13))

# condition = q_star > 0.99999

# q_star_refined = np.where(condition, 0.0, q_star)
# mass_evol_refined = np.where(condition, 0.0, mass_evol)

# q_star_refined = np.clip(q_star_refined, 0, 0.99999)
# mod_y = -np.log10(1-q_star_refined)

# def Temp(M, Q):
#     # if Q >= M/M_pl: return 0.0
#     kappa = np.sqrt(M**2 - (Q*M_pl)**2) / (2*M*(M + np.sqrt(M**2 - (Q*M_pl)**2)) - (Q*M_pl)**2)
#     return kappa / (2 * np.pi)

# M_g = mass_evol_refined / g_to_GeV

# plt.figure(figsize=(8, 6))
# plt.loglog(t_yr, M_g, 'b-', label=r'RN+BSM ($M_{PBH}$)')
# plt.loglog(t_yr, Temp(mass_evol_refined, q_star_refined*mass_evol_refined/M_pl), 'y-', label=r'RN+BSM ($M_{PBH}$)')


# ax1 = plt.gca()
# ax2 = ax1.twinx()
# ax2.semilogx(t_yr, mod_y, 'r-', label=r'Charge $-Log(1-Q/M)$')

# ax1.set_xlabel("t [yr]")
# ax1.set_ylabel(r"$M_{PBH}$ [g]", color='b')
# ax2.set_ylabel(r"$-Log_{10}[1 - Q^*_D]$", color='r')

# ax1.axvline(380000, color='grey', linestyle='-', linewidth=0.5, alpha=0.5) 
# ax1.axvline(1.38e10, color='orange', linestyle='-', linewidth=1.5, label=r'$t_{universe}$')

# ax1.set_ylim(1e8, 2e14)
# ax1.set_xlim(1e-1, 2e14) 

# plt.title("graph")
# plt.show()


### MAKE THE CHANGES BELOW!!!!!

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as scp
import scipy.integrate as sci
from scipy.integrate import solve_ivp
import pylab

e = 1.38*10**(-36)
me = 6.75*10**(-58)
Qo = 2.5*10**(8)
mD = 10**(10)
eD = 3*10**(-4)
M_pl = 1.22*10**(19)
g_to_GeV = 5.61*10**(23)
GeV_to_sec = 6.5821e-25    
sec_to_yr  = 1.0 / (365.25*24*3600)
hc = 2.6*10**(-70)
alpha1 = 2.0228
alpha2 = 0.2679
Mo = 1477

# alpha_rough = np.genfromtxt("/home/sarthaka/Downloads/blackhawk_v2.3/scripts/greybody_scripts/greybody_factors/charged/alpha_charged.txt")

# def inter_func(file):
#     some_variable = scp.interp1d(file[:,0], file[:,1], kind='cubic', bounds_error=False, fill_value=0.0)
#     return some_variable

# def alpha_smooth(M, Q):
#         q = M_pl*abs(Q/M)
#         alpha_inter = inter_func(alpha_rough)
#         val = alpha_inter(q)
#         return val


### for the visualization of alpha_smooth
### start
# M = 9.6*10**(12)*g_to_GeV
# Q = 0.01*M/M_pl
# x_data = np.linspace(0,0.9999,10000)
# y_data = [alpha_smooth(M, q*M) for q in x_data]
# fig = plt.figure(figsize=(8, 6))
# plt.plot(x_data, y_data, 'b-')
# plt.show()
### end

def rplus(M, Q):
    return M*(1+np.sqrt(1-(Q/M)**2))


# def de_system(t_sec, y):
#     M, Q = y
#     q = M_pl*abs(Q/M)
#     if q >= 0.85:
#         dQdt_gev = 10**(-28.135)
#         alpha = 10**(-28.135)
#     else:
#         term = eD**4*abs(Q)**3/(2*mD**2*np.pi**3*rplus(M, Q)**3)
#         dQdt_gev = -10**(-100)*term*np.exp(-np.pi*rplus(M,Q)**2*mD**2/(eD*abs(Q)))
#         alpha = alpha_smooth(M, Q)
         
#     alpha_term = -alpha*M_pl**4/M**2
#     another_term = Q*dQdt_gev/rplus(M,Q)
#     dMdt_gev = alpha_term + another_term
#     dQdt_sec = dQdt_gev / GeV_to_sec
#     dMdt_sec = dMdt_gev / GeV_to_sec
#     return [dMdt_sec, dQdt_sec]

# def de_system(t_sec, y):
#     M, Q = y
#     q = M_pl*abs(Q/M)
#     if q >= 0.85:
#         dQdt_gev = 0.0001
#         alpha = 0
#     if abs(Q) < 10**(-1000): 
#         dQdt_gev = 0.0
#     else:
#         alpha = alpha_smooth(M, Q)
#         arg = np.pi*rplus(M,Q)**2*mD**2/(eD*abs(Q)) 
#         if arg > 10**(20):
#             dQdt_gev = 0.0
#         else:
#             term = eD**4*abs(Q)**3/(2*mD**2*np.pi**3*rplus(M, Q)**3)
#             dQdt_gev = -term*np.exp(-arg)

#     alpha_term = -alpha*M_pl**4/M**2    
#     another_term = Q*dQdt_gev/rplus(M,Q)
#     dMdt_gev = alpha_term + another_term
#     dQdt_sec = dQdt_gev / GeV_to_sec
#     dMdt_sec = dMdt_gev / GeV_to_sec
#     return [dMdt_sec, dQdt_sec]

# def de_system(t_sec, y):
#     M, Q = y
#     q = M_pl*abs(Q/M)
#     arg = np.pi*rplus(M,Q)**2*mD**2/(eD*abs(Q)) 
#     term = eD**4*abs(Q)**3/(2*mD**2*np.pi**3*rplus(M, Q)**3)
#     if q >= 0.88:
#          dQdt_gev = 1e-30
#          dMdt_gev = 1e-30
#     else:
#         dQdt_gev = -term*np.exp(-arg)
#         alpha = alpha_smooth(M, Q)
#         alpha_term = -alpha*M_pl**4/M**2    
#         another_term = Q*dQdt_gev/rplus(M,Q)
#         dMdt_gev = alpha_term + another_term

#     dQdt_sec = dQdt_gev / GeV_to_sec
#     dMdt_sec = dMdt_gev / GeV_to_sec
#     return [dMdt_sec, dQdt_sec]
from scipy import special

# def de_system(t_sec, y):
#     M, Q = y
#     Q = max(Q, 1e-100)
#     q = M_pl*abs(Q/M)
#     arg = np.pi*rplus(M,Q)**2*mD**2/(eD*abs(Q)) 
#     term = eD**4*abs(Q)**3/(2*mD**2*np.pi**3*rplus(M, Q)**3)
#     log_prefactor = 4*np.log10(eD) + 3*np.log10(Q) - (np.log10(2) + 2*np.log10(mD) + 3*np.log10(np.pi) + 3*np.log10(rplus(M, Q)))
#     if log_prefactor > 50: # 10^50 GeV is enormous
#         log_prefactor = 50
        
#     term = 10**log_prefactor
#     if arg > 150:
#          term1 = 0.0
#          term2 = 0.0
#     else: 
#          term1 = -term*np.exp(-arg)
#          term2 = -special.erfc(arg**(0.5))*(np.pi**(1.5)*mD/(eD**(0.5)*abs(Q)**(0.5)))
#     dQdt_gev = term1 + term2
#     alpha = alpha_smooth(M, Q)
#     alpha_term = -alpha*M_pl**4/M**2    
#     another_term = Q*dQdt_gev/rplus(M,Q)
#     dMdt_gev = alpha_term + another_term

#     dQdt_sec = dQdt_gev / GeV_to_sec
#     dMdt_sec = dMdt_gev / GeV_to_sec
#     return [dMdt_sec, dQdt_sec]

def Temp(M, Q):
    # if Q >= M/M_pl: return 0.0
    kappa = np.sqrt(M**2-Q**2)/(rplus(M,Q)**2)
    return kappa*hc/(2 * np.pi)

def de_system(t_sec, y):
    M, Q = y
    arg = rplus(M,Q)**2/Q*Qo
    term = eD**3/(np.pi**2*hc**2*rplus(M,Q))
    term1 = -term*np.exp(-arg)
    term2 = -special.erfc(arg**(0.5))*(np.pi/(np.sqrt(Q*Qo)))
    dQdt = term1 + term2
    sigma = np.pi*((3*M+np.sqrt(9*M**2-8*Q**2))**4)/(8*(3*M**2-2*Q**2+M*np.sqrt(9*M**2-8*Q**2)))
    term3 = Temp(M,Q)**4*alpha1*sigma*np.pi**2/(15*hc**3)
    another_term = Q*dQdt/rplus(M,Q)
    dMdt = -term3 + another_term

    return [dMdt, dQdt]

# def arg(M,Q):
#     return rplus(M,Q)**2/Q*Qo

# def rate_of_change(y):
#     M, Q = y
#     q = M_pl*abs(Q/M)
#     alpha = alpha_smooth(M, Q)
#     arg_val = arg(M,Q)
#     term = eD**4*abs(Q)**3/(2*mD**2*np.pi**3*rplus(M, Q)**3)
#     dQdt_gev = -term*np.exp(-arg_val)

#     alpha_term = -alpha*M_pl**4/M**2    
#     another_term = Q*dQdt_gev/rplus(M,Q)
#     dMdt_gev = alpha_term + another_term
#     dQdt_sec = dQdt_gev / GeV_to_sec
#     dMdt_sec = dMdt_gev / GeV_to_sec
#     return [dMdt_sec, dQdt_sec]

mass_initial = 1.68*10**(8)*Mo
real_q = np.sqrt(0.1)*mass_initial
y0 = [mass_initial, real_q]


t_min_yr = float(1e-8)   
t_max_yr = float(10**(18))
t_eval_yr = np.geomspace(t_min_yr, t_max_yr, 100000)
t_eval_sec = t_eval_yr / sec_to_yr

sol = solve_ivp(de_system, [t_eval_sec[0], t_eval_sec[-1]], y0, 
                method='Radau', rtol=1e-6, atol=1e-10)


t_yr = sol.t * sec_to_yr
mass_evol = sol.y[0]
charge_evol = sol.y[1]

# q_star = charge_evol * M_pl / (mass_evol)

# condition = q_star > 0.99999

# q_star_refined = np.where(condition, 0.0, q_star)
# mass_evol_refined = np.where(condition, 0.0, mass_evol)

# q_star_refined = np.clip(q_star_refined, 0, 0.99999)
# mod_y = -np.log10(1-q_star_refined)


# M_g = mass_evol_refined / g_to_GeV

# plt.figure(figsize=(8, 6))
# plt.loglog(t_yr, M_g, 'b-', label=r'RN+BSM ($M_{PBH}$)')
# plt.loglog(t_yr, Temp(mass_evol_refined, q_star_refined*mass_evol_refined/M_pl), 'y-', label=r'RN+BSM ($M_{PBH}$)')


# ax1 = plt.gca()
# ax2 = ax1.twinx()
# ax2.semilogx(t_yr, mod_y, 'r-', label=r'Charge $-Log(1-Q/M)$')

# ax1.set_xlabel("t [yr]")
# ax1.set_ylabel(r"$M_{PBH}$ [g]", color='b')
# ax2.set_ylabel(r"$-Log_{10}[1 - Q^*_D]$", color='r')

# ax1.axvline(380000, color='grey', linestyle='-', linewidth=0.5, alpha=0.5) 
# ax1.axvline(1.38e10, color='orange', linestyle='-', linewidth=1.5, label=r'$t_{universe}$')

# ax1.set_ylim(1e8, 2e14)
# ax1.set_xlim(1e-1, 2e14) 

# plt.title("graph")
# plt.show()


f = pylab.figure(1)
f.clf()
ax = f.add_subplot(111)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("$t [yr]$")
ax.set_ylabel(r"M", color='b')
ax.plot(t_yr, mass_evol, 'b-', label=r'RN+BSM ($M_{PBH}$)')
f.show()

f = pylab.figure(2)
f.clf()
ax = f.add_subplot(111)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("$t [yr]$")
ax.set_ylabel(r"Q", color='r')
ax.plot(t_yr, charge_evol, 'r-', label=r'RN+BSM ($M_{PBH}$)')
f.show()

# f = pylab.figure(3)
# f.clf()
# ax = f.add_subplot(111)
# ax.set_xscale("log")
# ax.set_yscale("log")
# ax.set_xlabel("$t [yr]$")
# ax.set_ylabel(r"Temp [eV]", color='g')
# ax.plot(t_yr, Temp(mass_evol_refined, q_star_refined*mass_evol_refined/M_pl), 'g-', label=r'RN+BSM ($M_{PBH}$)')
# f.show()

# f = pylab.figure(4)
# f.clf()
# ax = f.add_subplot(111)
# ax.set_xscale("log")
# ax.set_yscale("log")
# ax.set_xlabel("$t [yr]$")
# ax.set_ylabel(r"Radius [GeV$^{-1}$]", color='y')
# ax.plot(t_yr, rplus(mass_evol, charge_evol), 'y-', label=r'RN+BSM ($M_{PBH}$)')
# f.show()


# #### below code is for phase space visualization

# # M_vect, Q_vect = np.meshgrid(mass_evol, charge_evol)
# # dM, dQ = rate_of_change([M_vect, Q_vect])
# # f = pylab.figure(6)
# # f.clf()
# # ax = f.add_subplot(111)
# # ax.set_xscale("log")
# # ax.set_yscale("log")
# # ax.set_xlabel("$M [g]$")
# # ax.set_ylabel(r"$Q\times M_pl$")
# # # plt.ylim(1e30, 1e36)
# # # plt.xlim(1e30, 1e36)
# # plt.quiver(M_vect, Q_vect*M_pl, dM, dQ*M_pl, angles='xy', scale_units='xy', scale=1)
# # plt.title(r'Phase Space for Q*M_pl-M')
# # plt.plot(mass_evol, mass_evol, 'r--')
# # ax.fill_between(mass_evol, 0, mass_evol, color='green', alpha=0.1, label='Allowed Region')
# # ax.fill_between(mass_evol, mass_evol, 1e35, color='red', alpha=0.1, label='Forbidden Region')
# # f.show()


# #### Corrected Phase Space Visualization

# f = pylab.figure(5)
# f.clf()
# ax = f.add_subplot(111)

# # --- LAYER 1: The Background Vector Field (The "Map") ---
# # Create a CLEAN, independent grid for the arrows
# # We span from slightly below your min mass to slightly above your max mass
# M_grid_vals = np.geomspace(min(mass_evol)/2, max(mass_evol)*2, 20)
# Q_grid_vals = np.geomspace(min(charge_evol)/2, max(charge_evol)*2, 20)

# MM, QQ = np.meshgrid(M_grid_vals, Q_grid_vals)

# # Calculate rates at these grid points
# # Note: We must handle the vectorization carefully for the rate function
# # We simply reuse your de_system logic but applied to the grid
# q_grid = M_pl * abs(QQ/MM)
# rplus_grid = MM*(1+np.sqrt(1-np.clip((q_grid)**2, 0, 1)))/M_pl**2 # Simplified r+ for grid vis
# arg_grid = np.pi * rplus_grid**2 * mD**2 / (eD * abs(QQ) + 1e-100)
# term_grid = eD**4 * abs(QQ)**3 / (2 * mD**2 * np.pi**3 * rplus_grid**3 + 1e-100)

# # Compute dQ and dM for the grid arrows
# dQ_grid = -term_grid * np.exp(-arg_grid)
# # (Approximate alpha for grid visualization to avoid heavy interpolation calls)
# alpha_grid = alpha_smooth(MM, QQ) # Just for visual flow direction, or use your interpolation if fast enough
# dM_grid = (-alpha_grid * M_pl**4 / MM**2) + (QQ * dQ_grid / rplus_grid)

# # Normalize arrows so they are visible (optional, prevents huge arrows)
# arrow_speed = np.sqrt(dM_grid**2 + dQ_grid**2)
# dM_norm = dM_grid / (arrow_speed)  # Avoid division by zero
# dQ_norm = dQ_grid / (arrow_speed)


# # Plot the arrows
# # We use log-log scale, so 'quiver' can be tricky.
# # Often simple streamplot is better, but quiver works if we are careful.
# ax.quiver(MM, QQ*M_pl, dM_norm, dQ_norm*M_pl, color='gray', alpha=0.3, pivot='mid')


# # --- LAYER 2: The Actual Trajectory (Your Solution) ---
# # Just plot the line! No meshgrid needed for the solution.
# ax.plot(mass_evol, charge_evol*M_pl, 'b-', linewidth=2, label='Evolution Path')

# # Mark the Start and End
# ax.plot(mass_evol[0], charge_evol[0]*M_pl, 'go', label='Start')
# ax.plot(mass_evol[-1], charge_evol[-1]*M_pl, 'rx', label='End')


# # --- Styling ---
# ax.set_xscale("log")
# ax.set_yscale("log")
# ax.set_xlabel(r"$M_{PBH}$ [GeV]")
# ax.set_ylabel(r"$Q \times M_{pl}$ [GeV]")
# ax.set_title(r'Evolution Trajectory for arg $> 80$')

# # Regions
# # Extremality Line Q*M_pl = M_BH implies Q*M_pl/M_BH = 1.
# # So in this plot of (M) vs (Q*M_pl), the line is y = x (in appropriate units)
# # Since we plot Q*M_pl on Y and M on X, the extremality line is Y = M (in Planck units conversion)
# # Ideally, just fill the forbidden region:
# x_fill = np.geomspace(min(mass_evol)/10, max(mass_evol)*10, 100)
# ax.fill_between(x_fill, x_fill, max(charge_evol*M_pl)*100, color='red', alpha=0.1, label=r'Forbidden ($Q*M_{pl} > M_{PBH}$)')
# ax.fill_between(x_fill, min(charge_evol*M_pl)/100, x_fill, color='green', alpha=0.1, label=r'Allowed ($Q*M_{pl} < M_{PBH}$)')

# ax.set_xlim(min(mass_evol)/1.5, max(mass_evol)*1.5)
# ax.set_ylim(min(charge_evol*M_pl)/1.5, max(charge_evol*M_pl)*1.5)
# ax.legend()
# f.show()

# f = pylab.figure(6)
# f.clf()
# ax = f.add_subplot(111)
# ax.set_xscale("log")
# ax.set_yscale("log")
# ax.set_xlabel("$t [yr]$")
# ax.set_ylabel("Arg of exponential", color='c')
# ax.plot(t_yr, arg(mass_evol, charge_evol), 'c-', label=r'RN+BSM ($M_{PBH}$)')
# f.show()

# f = pylab.figure(7)
# f.clf()
# ax = f.add_subplot(111)
# ax.set_xscale("log")
# ax.set_yscale("log")
# ax.set_xlabel("$t [yr]$")
# ax.set_ylabel(r"$dQ/dt$ [GeV/s]", color='m')
# ax.plot(t_yr, [de_system(t, [m, q])[1] for t, m, q in zip(sol.t, mass_evol, charge_evol)], 'm-', label=r'RN+BSM ($M_{PBH}$)')
# f.show()

# f = pylab.figure(8)
# f.clf()
# ax = f.add_subplot(111)
# # ax.set_xscale("log")
# # ax.set_yscale("log")
# ax.set_xlabel("$t [yr]$")
# ax.set_ylabel(r"$dM/dt$ [GeV/s]", color='m')
# # plt.xlim(3.246e6, 3.248e6)
# # plt.ylim(-1, 0)
# dMdt_smooth = np.log((-1)*[de_system(t, [m, q])[0] for t, m, q in zip(sol.t, mass_evol, charge_evol)])
# ax.plot(t_yr, dMdt_smooth, 'm-', label=r'RN+BSM ($M_{PBH}$)')
# f.show()
# #### parameters are as per the correct physics/trajectory is in the allowed region to avoid naked singularity

input("Press Enter to continue...")
pylab.close("all")

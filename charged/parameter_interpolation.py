import numpy as np
import scipy.interpolate as scp
import scipy.integrate as sci

fits_0 = np.genfromtxt("/home/sarthaka/Downloads/blackhawk_v2.3/scripts/greybody_scripts/greybody_factors/charged/fits_charged_spin_0.txt", skip_header=1)
fits_1 = np.genfromtxt("/home/sarthaka/Downloads/blackhawk_v2.3/scripts/greybody_scripts/greybody_factors/charged/fits_charged_spin_1.txt", skip_header=1)
fits_2 = np.genfromtxt("/home/sarthaka/Downloads/blackhawk_v2.3/scripts/greybody_scripts/greybody_factors/charged/fits_charged_spin_2.txt", skip_header=1)
fits_12 = np.genfromtxt("/home/sarthaka/Downloads/blackhawk_v2.3/scripts/greybody_scripts/greybody_factors/charged/fits_charged_spin_12.txt", skip_header=1)    

def make_interpolations(fits):
    qs = fits[:,0]
    a_cont = scp.interp1d(qs, fits[:,1], kind="cubic", fill_value="extrapolate")
    b_cont = scp.interp1d(qs, fits[:,2], kind="cubic", fill_value="extrapolate")
    lim_cont = scp.interp1d(qs, fits[:,3], kind="cubic", fill_value="extrapolate")
    return a_cont, b_cont, lim_cont

cont_func_0 = make_interpolations(fits_0)
cont_func_1 = make_interpolations(fits_1)
cont_func_2 = make_interpolations(fits_2)
cont_func_12 = make_interpolations(fits_12)

M = 1

def distribution(E, M, Q, const_funcs):
    q = Q/M
    if q>0.999: q=0.999

    a_param = const_funcs[0](q)
    b_param = const_funcs[1](q)
    lim_param = const_funcs[2](q)
    
    x = E*M

    flux_low = (x**(a_param))*10**(b_param)
    flux_high = lim_param*(27./4.)*x**2

    return np.maximum(flux_low, flux_high)

def Temp(M, Q):
    if Q >= M: return 0.0
    r_plus = M + np.sqrt(M**2 - Q**2)
    kappa = np.sqrt(M**2 - Q**2) / (2*M*(M + np.sqrt(M**2 - Q**2)) - Q**2)
    return kappa / (2 * np.pi)

def integrand(E, M, Q):
    if Temp(M, Q) == 0.0:
        return 0.0
    term0 = distribution(E, M, Q, cont_func_0)/(np.exp(E/Temp(M, Q)) - 1)
    term1 = distribution(E, M, Q, cont_func_1)/(np.exp(E/Temp(M, Q)) - 1)
    term2 = distribution(E, M, Q, cont_func_2)/(np.exp(E/Temp(M, Q)) - 1)
    term12 = distribution(E, M, Q, cont_func_12)/(np.exp(E/Temp(M, Q)) + 1)
    return (term0 + term1 + term2 + term12)*E/(2*np.pi)


def alpha(Q):
    lim_low = 10**(-10)*Temp(M, Q)
    lim_high = 10**2*Temp(M, Q)
    return sci.quad(integrand, lim_low, lim_high, args=(M,Q))[0]

q_array = np.linspace(0,1,10000)
results = []

for q in q_array:
    if q>=1:
        alpha(q) == 0.0
    else:
        results.append([q, alpha(q)])

np.savetxt("/home/sarthaka/Downloads/blackhawk_v2.3/scripts/greybody_scripts/greybody_factors/charged/alpha_charged.txt", results)
print("Done")

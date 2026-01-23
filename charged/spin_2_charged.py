import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import lambertw
from scipy.optimize import fsolve



nbmodes = 30
nbener = 50
M = 0.5
s = 2
nbrQ = 50
rQmin = 0.01
rQmax = 0.999

iteration_array1 = np.arange(0, nbrQ, 1)
iteration_array2 = np.arange(0, nbener, 1)
iteration_array3 = np.arange(0, nbmodes, 1)

rQ = []

for i in iteration_array1:
    imp_function = (1 - 10**(np.log10(1 - rQmin) + (np.log10(1 - rQmax) - np.log10(1 - rQmin))/(nbrQ - 1)*i))*M
    rQ.append(imp_function)

## un-comment the below energy values and omega function and comment the more below for low energy  

Emin = 10**(-6) 
Emax = 10**(-4) 

omega = []

for i in iteration_array2:
    energy_function = 10**(np.log10(Emin) + (np.log10(Emax)-np.log10(Emin))/(nbener - 1)*i)
    omega.append(energy_function)
 
## un-comment the below energy values and omega function and comment the above for high energy  

# Emin = 10**(-2)
# Einter = 1
# Emax = 5

# omega = []

# for i in iteration_array2:
#     if i < 100:
#         energy_function = 10**(np.log10(Emin) + (np.log10(Einter) - np.log10(Emin))/100*i)
#     else:
#         energy_function = Einter + (Emax - Einter)/(99)*(i - 100)
#     omega.append(energy_function)   


l = []
for i in iteration_array3:
    l.append(i+2)

Ainh = 1

def nu_2(l):
    return l*(l + 1) - 2

def rplus(M, rQ):
    return M*(1 + np.sqrt(1 - 4*rQ**2/(4*M**2)))

def rminus(M, rQ):
    return M*(1 - np.sqrt(1 - 4*rQ**2/(4*M**2)))

def rstar(r, M, rQ):
    return r + rplus(M, rQ)**2/(rplus(M, rQ) - rminus(M, rQ))*np.log(r/rplus(M, rQ) - 1) - rminus(M, rQ)**2/(rplus(M, rQ) - rminus(M, rQ))*  np.log(r/rminus(M, rQ) - 1)

def invertr(rS, M, rQ):
    if rQ == 0:
        arg = np.exp(rS / (2 * M) - 1)
        val = np.real(lambertw(arg))
        return 2 * M * (1 + val)
    else:
        if rS <= 5:
            start = rplus(M, rQ) + 0.00001
        else:
            start = rS
        def func(u):
            return rstar(u, M, rQ) - rS
        return fsolve(func, start)[0]

def H(r):
    return r**2

def F(r, M, rQ):
    return 1-2*M/r + rQ**2/r**2

def dF_dr(r, M, rQ):
    return 2*M/r**2 - 2*rQ**2/r**3

def G(r, M, rQ):
    return 1-2*M/r + rQ**2/r**2

def V1(r, M, rQ, l):
    return nu_2(l)*G(r, M, rQ)/H(r) + 2*F(r, M, rQ)**2/r**2 - F(r, M, rQ)*dF_dr(r, M, rQ)/r


printsteps = 0
prox = 0.0000000001
far = 10000000  # for low energy
# far = 500     # for high energy

def solver(M, rQ, omega, l):
    rp = rplus(M, rQ)
    rm = rminus(M, rQ)
    rminf = rstar(rp + prox, M, rQ)
    rpinf = rstar(rp+far, M, rQ)
    r_init = rp + prox  
    phase = -omega * rminf
    X_val = Ainh * np.exp(1j * phase)
    dX_val = -1j * omega * X_val

    y0 = [r_init, np.real(X_val), np.imag(X_val), np.real(dX_val), np.imag(dX_val)]

    def system(t, y):
        r_curr, X_re, X_im, dX_re, dX_im = y
        
        dr_drs = (r_curr - rp) * (r_curr - rm) / (r_curr**2)
        
        pot = V1(r_curr, M, rQ, l)
        factor = pot - omega**2
        
        ddX_re = factor * X_re
        ddX_im = factor * X_im
        
        return [dr_drs, dX_re, dX_im, ddX_re, ddX_im]

    sol = solve_ivp(system, (rminf, rpinf), y0, method='LSODA', rtol=1e-12, atol=1e-12)
    
    final_rs = sol.t[-1]      
    final_r_ode = sol.y[0][-1]

    return sol

file1 = open("/home/sarthaka/Downloads/blackhawk_v2.3/scripts/greybody_scripts/greybody_factors/charged/fM_charged_2_low_py.txt", "w")
## change the name low to high depending on the energy value


s0      = [[[[None for _ in range(2 * l[i] + 1)] for i in range(nbmodes)] for j in range(nbener)] for k in range(nbrQ)]
Ain0    = [[[[0j   for _ in range(2 * l[i] + 1)] for i in range(nbmodes)] for j in range(nbener)] for k in range(nbrQ)]
Aout0   = [[[[0j   for _ in range(2 * l[i] + 1)] for i in range(nbmodes)] for j in range(nbener)] for k in range(nbrQ)]
contrib = [[[[0.0  for _ in range(2 * l[i] + 1)] for i in range(nbmodes)] for j in range(nbener)] for k in range(nbrQ)]

Al0 = np.zeros((nbrQ, nbener))

optimize = True
print_debug = False

def Temp(M, rQ):
    return (rplus(M, rQ) - rminus(M, rQ))/(4*np.pi*rplus(M, rQ)**2)

for k in range(nbrQ):

    for j in range(nbener):

        Al0[k, j] = 0.0

        for i in range(nbmodes):

            if optimize and i>0 and Al0[k,j]>10**(-100):
                max_contrib1 = max(contrib[k][j][i-1])
                if abs(max_contrib1/Al0[k,j])<10**(-5):
                    break
            for m in range(-l[i], l[i]+1):
                if m>-l[i] and optimize and Al0[k,j]>10**(-100):
                    contrib2 = contrib[k][j][i][m+l[i]-1]
                    if abs(contrib2/Al0[k,j])<10**(-5):
                        break
                if m>-l[i]:
                    s0[k][j][i][m+l[i]] = s0[k][j][i][m+l[i]-1]
                else: 
                    s0[k][j][i][m+l[i]] = solver(M, rQ[k], omega[j], l[i])

                rpinf = rstar(rplus(M, rQ[k]) + far, M, rQ[k])
                sol_obj = s0[k][j][i][m+l[i]]
                
                X_val  = sol_obj.y[1][-1] + 1j * sol_obj.y[2][-1]
                dX_val = sol_obj.y[3][-1] + 1j * sol_obj.y[4][-1]
                Ain0[k][j][i][m+l[i]] = (X_val + dX_val/(1j * omega[j])) / (2 * np.exp(1j * omega[j] * rpinf))
                Aout0[k][j][i][m+l[i]] = (X_val - dX_val/(1j * omega[j])) / (2 * np.exp(-1j * omega[j] * rpinf))

                T = Temp(M, rQ[k])
                if T < 1e-9:
                    be_factor = 0.0
                else:
                    try:
                        be_factor = 1.0 / (np.exp(omega[j] / T) - 1)
                    except OverflowError:
                        be_factor = 0.0
                
                term = (1.0 / abs(Aout0[k][j][i][m+l[i]])**2) * be_factor
                contrib[k][j][i][m+l[i]] = term
                
                Al0[k, j] = Al0[k, j] + contrib[k][j][i][m+l[i]]
                
                if m == -l[i]:
                    pass
    line_data = [f"{val:.6e}" for val in Al0[k]]
    file1.write("\t".join(line_data) + "\n")
    
    file1.flush()

file1.close()
print("DONE")
    
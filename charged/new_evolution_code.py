import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

# ─────────────────────────────────────────────────────────────────
#  All quantities in geometrodynamic units: G = c = 1, unit = metre
# ─────────────────────────────────────────────────────────────────
hbar    = 2.6e-70     # ħ              [m²]
e_std   = 1.38e-36    # electron charge [m]
me_std  = 6.75e-58    # electron mass   [m]
M_sun_m = 1477.0      # 1 M☉           [m]
alpha   = 0.2679      # α (0 massless neutrinos)


# ─────────────────────────────────────────────────────────────────
#  Dark electron: your mD, eD reinterpreted as dimensionless
#  scaling factors  mχ = σ_m · mₑ,  eχ = σ_e · e
# ─────────────────────────────────────────────────────────────────
sigma_m =  1   ## 1e10
sigma_e =  1   ## 3e-4

mchi = sigma_m * me_std    # [m]
echi = sigma_e * e_std     # [m]

# Charge-mass scale  [Eq. 9 in Santiago+]
Q0 = hbar * echi / (np.pi * mchi**2)
print(f"Q₀ = {Q0:.3e} m  =  {Q0/M_sun_m:.3e} M☉")

# ─────────────────────────────────────────────────────────────────
#  Dimensionless parameters z₀, b₀, s₀  [Santiago+ Eqs. 22a-c]
# ─────────────────────────────────────────────────────────────────
def compute_params(Ms):
    b0  = Ms / Q0
    arg = 960.0 * echi**4 * Ms**2 / (np.pi**2 * alpha * mchi**2 * hbar**2)
    z0  = (Q0 / Ms) * np.log(arg) if arg > 1.0 else 0.0
    s0  = alpha * hbar / (1920.0 * np.pi * Ms**2)
    return z0, b0, s0

# ─────────────────────────────────────────────────────────────────
#  Find Mₛ so that z₀ ≈ 0.53  (same normalisation as HW/Santiago+)
#  brentq finds the descending branch (large Mₛ) where b₀ ≫ 1
# ─────────────────────────────────────────────────────────────────
log_Ms_sol = brentq(
    lambda lMs: compute_params(10**lMs)[0] - 0.52682,
    -14.0, 25.0          # log₁₀(Mₛ / m) search range
)
Ms = 10**log_Ms_sol
z0, b0, s0 = compute_params(Ms)

zchi_0 = echi*hbar*np.log(960*echi**4*Ms**2/(np.pi**2*alpha*mchi**2*hbar**2)) / (np.pi*mchi**2*Ms)  # [m]

print(f"Mₛ  = {Ms:.3e} m  =  {Ms/M_sun_m:.3e} M☉")
print(f"z₀  = {z0:.4f},   b₀ = {b0:.1f}")
print(f"Mz₀ = {z0*Ms/M_sun_m:.3e} M☉   ← attractor mass scale")

# ─────────────────────────────────────────────────────────────────
#  ODE right-hand side in (μ, Y) space  [Santiago+ Eqs. 23-24]
#
#  KEY: S(μ,Y) is reformulated so the exponent is relative to z₀,
#  not an absolute huge negative number → no floating-point underflow
# ─────────────────────────────────────────────────────────────────
def rhs(tau, state, b0, z0):    
    mu, Y = state
    if mu <= 0.0 or Y <= 0.0 or Y >= 1.0:
        return [0.0, 0.0]

    sqY  = np.sqrt(Y)
    s1mY = np.sqrt(1.0 - Y)
    s98Y = np.sqrt(max(9.0 - 8.0*Y, 0.0))

    # H(μ, Y)  [Eq. 20] — Hawking term
    H = ((s98Y + 3.0)**4 * (1.0 - Y)**2
         / (mu**2 * (s1mY + 1.0)**4 * (3.0 - 2.0*Y + s98Y)))

    # S(μ, Y)  [Eq. 21] — Schwinger term, clipped to avoid overflow
    #   In mass dissipation zone: exponent ≪ 0  →  S ≈ 0  (correct: charge frozen)
    #   On attractor:             exponent = 0  →  S = 1
    #   In charge dissipation:    exponent ≫ 0  →  S ≫ 1  (fast discharge)
    exp_arg = b0 * (z0 - mu * (s1mY + 1.0)**2 / sqY)
    S = np.exp(np.clip(exp_arg, -500.0, 500.0))

    D = (s1mY + 1.0)**4

    dmu = -(H + S * Y**2) / D                               # Eq. 23
    dY  =  2.0 * Y * (H - S*(1.0 - Y + s1mY)*Y) / (mu * D) # Eq. 24
    return [dmu, dY]


# ─────────────────────────────────────────────────────────────────
#  Approximate attractor curve  [Eq. 30]:  μ = z₀√Y / (√(1-Y)+1)²
# ─────────────────────────────────────────────────────────────────
def attractor_Y(mu, z0):
    if mu >= z0:
        return np.nan
    try:
        return brentq(
            lambda Y: mu - z0*np.sqrt(max(Y, 1e-30))
                              / (np.sqrt(max(1.0-Y, 0.0)) + 1.0)**2,
            1e-12, 1.0 - 1e-12, xtol=1e-12
        )
    except Exception:
        return np.nan

mu_att = np.linspace(1e-4, 0.9999*z0, 400)
Y_att  = np.array([attractor_Y(m, z0) for m in mu_att])

# ─────────────────────────────────────────────────────────────────
#  Initial conditions — both dissipation zones
# ─────────────────────────────────────────────────────────────────
# ics = [
#     # mass dissipation zone (low Y → charge-to-mass ratio small)
#     (1.00, 0.60), (1.00, 0.325), (1.00, 0.20), (1.00, 0.15), (1.00, 0.05), (1.00, 0.01), (1.00, 0.00),
#     # charge dissipation zone (high Y → near extremal)
#     (0.40, 0.999), (0.30, .999), (0.20, 0.999), (0.10, 0.999),
# ]

ics = [(0.999, 0.5)]

mu_h_list = [0.8, 0.6, 0.37, 0.25, 0.18, 0.07]
mu_s_list = [0.4, 0.3, 0.2, 0.1]

# ─────────────────────────────────────────────────────────────────
#  Integrate
# ─────────────────────────────────────────────────────────────────
tau_max = 10**(45)

ev_mu = lambda t, y, *a: y[0] - 5e-4   # stop at μ → 0
ev_mu.terminal = True;  ev_mu.direction = -1

ev_Y  = lambda t, y, *a: 1.0 - y[1] - 1e-6   # stop at Y → 1
ev_Y.terminal = True;   ev_Y.direction = -1

# ─────────────────────────────────────────────────────────────────
#  Plot — reproducing HW Fig. 2 / Santiago+ Fig. 2 topology
# ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 7))

# Shaded zones (fill relative to attractor curve)
mu_f = np.linspace(1e-4, 0.9999, 500)
Y_f  = np.array([attractor_Y(m, z0) for m in mu_f])
Y_f  = np.where(np.isnan(Y_f), 1.0, Y_f)
ax.fill_between(mu_f, Y_f,  1.0,  alpha=0.12, color='green',
                label='Charge dissipation zone')
ax.fill_between(mu_f, 0.0,  Y_f,  alpha=0.12, color='royalblue',
                label='Mass dissipation zone')

# Attractor
ok = ~np.isnan(Y_att)
ax.plot(mu_att[ok], Y_att[ok], 'r--', lw=2.5, label='Approx. attractor', zorder=5)

# Trajectories
# for mu0, Y0 in ics:
#     try:
#         sol = solve_ivp(rhs, [0.0, tau_max], [mu0, Y0],
#                         method='Radau', args=(b0, z0),
#                         rtol=1e-10, atol=1e-12,
#                         max_step=tau_max/3000,
#                         events=[ev_mu, ev_Y])
#         ms_rhs, Ys_rhs = sol.y
#         ok = (ms_rhs > 0) & (Ys_rhs > 0) & (Ys_rhs <= 1.0)
#         ax.plot(ms_rhs[ok], Ys_rhs[ok], 'b-', lw=0.9, alpha=0.8)  # ← just plot directly
#         ax.plot(mu0, Y0, 'k.', ms=5, zorder=6)
#     except Exception as ex:
#         print(f"  IC ({mu0:.2f},{Y0:.2f}) failed: {ex}")

def config_space_ode(mu, Y_arr, b0, z0):
    Y = np.clip(Y_arr[0], 1e-10, 1.0 - 1e-10)   # clip instead of freeze
    if mu <= 0:
        return [0.0]

    s1mY = np.sqrt(max(1.0 - Y, 0.0))
    sqY  = np.sqrt(Y)
    s98Y = np.sqrt(max(9.0 - 8.0*Y, 0.0))

    H = ((s98Y + 3.0)**4 * (1.0 - Y)**2
         / (mu**2 * (s1mY + 1.0)**4 * (3.0 - 2.0*Y + s98Y)))

    exp_arg = b0 * (z0 - mu * (s1mY + 1.0)**2 / sqY)
    S = np.exp(np.clip(exp_arg, -500.0, 500.0))

    # Eq. 25 — note: μ is decreasing so solve_ivp integrates backward
    numerator = 2.0 * Y * (H - S * (1.0 - Y + s1mY) * Y)
    denominator = mu * (H + S * Y**2)
    
    dYdmu = -numerator / denominator
    return [dYdmu]

mu0, Y0 = 0.999, 0.5

ev_mu0 = lambda mu, Y, *a: mu - 1e-4
ev_mu0.terminal = True; ev_mu0.direction = -1  # μ → 0

sol1 = sol_config = solve_ivp(
    config_space_ode,
    [mu0, z0],          # μ decreases: t_span[0] > t_span[1]
    [Y0],
    method='Radau',
    args=(b0, z0),
    rtol=1e-10, atol=1e-12,
    max_step=1e-4,
    events=[ev_mu0],
    dense_output=True
)

sol2 = solve_ivp(
    config_space_ode, [z0 - 1e-6, 1e-4], [1.0 - 1e-6],
    method='Radau', args=(b0, z0),
    rtol=1e-10, atol=1e-12, max_step=1e-4,
    events=[ev_mu0]
)

mu_traj = np.concatenate([sol1.t, sol2.t])
Y_traj  = np.concatenate([sol1.y[0], sol2.y[0]])
Y_traj  = np.clip(Y_traj, 0.0, 1.0)

ax.plot(mu_traj, Y_traj, 'b-', lw=1.5, label=r'Numerical $Y(\mu)$', zorder=4)
ax.plot(mu0, Y0, 'ko', ms=6, zorder=7)
ax.set_xlim(0.0, 1.1 * mu0)

# def hawking_curve(ax, mu_h_list):
#     for mu_h in mu_h_list:
#       mu_haw = np.linspace(mu_h, 1, 2000)
#       func_H = (mu_h/mu_haw)**2
#       ax.plot(mu_haw, func_H, 'b--', lw=0.9, alpha=0.8)
#       ax.plot(1.0, func_H[-1], 'k.', ms=5, zorder=6)
# hawking_curve(ax, mu_h_list)


# def schwinger_curve(ax, mu_s_list):
#     for mu_s in mu_s_list:
#       mu_sch = np.linspace(1e-6, mu_s, 2000)
#       func_S = ((2*mu_sch-mu_s)*mu_s)/(mu_sch**2)
#       ax.plot(mu_sch, func_S, 'b--', lw=0.9, alpha=0.8)
#       ax.plot(mu_s, 1.0, 'k.', ms=5, zorder=6)
# schwinger_curve(ax, mu_s_list)

# Reference lines
ax.axhline(1.0, c='gray',   ls='--', lw=1.0, alpha=0.7, label='$Q=M$ (extremal)')
ax.axvline(z0,  c='salmon', ls=':',  lw=1.2, alpha=0.8,
           label=rf'$z_0 = {z0:.3f}$')
idx_peak = np.argmax(Y_traj)
ax.axvline(mu_traj[idx_peak], c='orange', ls=':', lw=1.2,
           label=rf'$\mu_h = {mu_traj[idx_peak]:.3f}$')

ax.text(0.13, 0.82, 'Charge\ndissipation\nzone',
        fontsize=10, color='darkgreen', ha='center')
ax.text(0.65, 0.10, 'Mass dissipation zone',
        fontsize=10, color='navy',      ha='center')

ax.set_xlim(0.0, 1.0)
ax.set_ylim(0.0, 1.05)
ax.set_xlabel(rf'$\mu = M/M_s$   ($M_s = {Ms/M_sun_m:.2e}\,M_\odot$)', fontsize=12)
ax.set_ylabel(r'$Y = (Q/M)^2$', fontsize=12)
ax.set_title(
    rf'($\sigma_m=1,\;\sigma_e=1$)' '\n'
    rf'$b_0={b0:.0f}$,   $M_{{z_0}} = {z0*Ms/M_sun_m:.2e}\,M_\odot$',
    fontsize=10)
ax.legend(fontsize=9, loc='center right')
ax.grid(True, ls='--', lw=0.4, alpha=0.4)

plt.tight_layout()
plt.show()

# ─────────────────────────────────────────────────────────────────
#  Helper: convert your original (M_GeV, Q_GeV) to (μ, Y)
# ─────────────────────────────────────────────────────────────────
GeV_to_m = 1.0 / (1.22e19 * 1.0)   # 1 GeV = 1/M_pl metres (geom units)

def physical_to_muY(M_GeV, Q_GeV, Ms):
    M_m = M_GeV * GeV_to_m          # convert to metres
    Q_m = Q_GeV * GeV_to_m
    mu  = M_m / Ms
    Y   = (Q_m / M_m)**2
    return mu, Y

# Example: your original IC
M0_GeV = 1e8 * M_sun_m / GeV_to_m  # 10^8 M_sun in GeV (just for demo)
# Note: μ for 10^8 M_sun will be >> 1 with these dark electron params,
# meaning you're many decades above the attractor → pure Hawking evaporation
# until M drops to ~ Mz0 ~ 10^-16 M_sun.
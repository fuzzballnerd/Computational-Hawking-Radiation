from scipy.interpolate import interp1d
from scipy.optimize import brentq
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt

# ── Constants ────────────────────────────────────────────────────
hbar    = 2.6e-70
e_std   = 1.38e-36
me_std  = 6.75e-58
M_sun_m = 1477.0
yr_to_s = 3.156e7
GeV_to_m= 1.0/1.22e19

sigma_m = 1;  sigma_e = 1
mchi = sigma_m * me_std
echi = sigma_e * e_std
Q0   = hbar * echi / (np.pi * mchi**2)

# ── User-facing initial conditions ───────────────────────────────
# The evolution is solved in μ = M / Ms and Y = (Q/M)^2. If you care
# about a specific physical mass, set it here and let μ₀ be inferred
# from the bootstrapped Ms below.
target_z0 = 0.52682  # z₀ on the attractor (Santiago+ Eq. 22c)
M0_target_Msun = 1e-20
Q_over_M0 = 0.999

# ── Load α(Q/M) table ────────────────────────────────────────────
data       = np.loadtxt('/home/batman/Downloads/blackhawk_v2.3/scripts/greybody_scripts/greybody_factors/charged/new_alpha_charged.txt')
QoM_vals   = data[:, 0]
alpha_vals = data[:, 1]
alpha_interp = interp1d(QoM_vals, alpha_vals, kind='cubic',
                        bounds_error=False,
                        fill_value=(alpha_vals[0], alpha_vals[-1]))

alpha_ref = float(alpha_interp(0.0))   # Schwarzschild limit
print(f"α_ref = {alpha_ref:.6e}")

def alpha_of_Y(Y):
    """α as function of Y=(Q/M)², via Q/M=√Y."""
    return float(alpha_interp(np.clip(np.sqrt(max(Y, 0.0)), 0.0, 1.0)))

# ── Ms bootstrap ─────────────────────────────────────────────────
def compute_params(Ms):
    b0  = Ms / Q0
    arg = 960.0*echi**4*Ms**2 / (np.pi**2*alpha_ref*mchi**2*hbar**2)
    z0  = (Q0/Ms)*np.log(arg) if arg > 1.0 else 0.0
    s0  = alpha_ref*hbar / (1920.0*np.pi*Ms**2)
    return z0, b0, s0

log_Ms_sol = brentq(lambda lMs: compute_params(10**lMs)[0] - target_z0,
                    -14.0, 25.0)
Ms = 10**log_Ms_sol
z0, b0, s0 = compute_params(Ms)
print(f"Ms={Ms/M_sun_m:.3e} M☉  z0={z0:.4f}  b0={b0:.1f}")
print(f"s0={s0:.3e}  Ms/s0 = {Ms/s0:.3e} m  = "
      f"{Ms/s0/(M_sun_m*yr_to_s*3e8):.3e} yr")
# Ms/s0 has units of [m/m² * m²] → seconds when divided by c=1

# physical time unit: t_phys = τ · (Ms/s0) in geometrodynamic seconds
# to years: divide by (yr_to_s * c_in_m_per_s) but in geom units c=1
# so: t[yr] = τ · (Ms/s0) / (c · yr_to_s) where c=3e8 m/s... 
# careful: in geom units [time]=[length]/c, so τ·Ms/s0 is in metres
# t[s] = τ·Ms/s0 / c = τ·Ms/(s0 * 3e8)
c_si = 3e8  # m/sscal
t_scale_yr = Ms / (s0 * c_si * yr_to_s)   # converts τ → years
print(f"τ→year scale: {t_scale_yr:.3e} yr per unit τ")

# ── H and S terms with α(Y) ──────────────────────────────────────
def H_and_S(mu, Y):
    s1mY = np.sqrt(max(1.0-Y, 0.0))
    sqY  = np.sqrt(max(Y,     0.0))
    s98Y = np.sqrt(max(9.0-8.0*Y, 0.0))

    # Full H(μ,Y) [Santiago Eq.20] × α(√Y)/α_ref
    H_geom = ((s98Y+3.0)**4 * (1.0-Y)**2
              / (mu**2 * (s1mY+1.0)**4 * (3.0-2.0*Y+s98Y)))
    H = (alpha_of_Y(Y) / alpha_ref) * H_geom

    # Schwinger S [Santiago Eq.21] — analytic; replace with table if available
    exp_arg = b0*(z0 - mu*(s1mY+1.0)**2/sqY)
    S = np.exp(np.clip(exp_arg, -500.0, 500.0))

    return H, S

# ── τ-time RHS  [Santiago Eqs. 23-24] ────────────────────────────
def rhs_tau(tau, state):
    mu, Y = state
    # guard
    mu = max(mu, 1e-12)
    Y  = np.clip(Y, 1e-12, 1.0-1e-12)

    s1mY = np.sqrt(1.0-Y)
    H, S = H_and_S(mu, Y)
    D    = (s1mY+1.0)**4

    dmu = -(H + S*Y**2) / D
    dY  =  2.0*Y*(H - S*(1.0-Y+s1mY)*Y) / (mu*D)
    return [dmu, dY]

# ── Attractor: numerical (dY/dτ=0 condition) ─────────────────────
def attractor_Y_num(mu):
    if mu >= z0: return np.nan
    def residual(Y):
        Y  = np.clip(Y, 1e-10, 1-1e-10)
        H, S = H_and_S(mu, Y)
        s1mY = np.sqrt(max(1-Y,0))
        return H - S*(1.0-Y+s1mY)*Y
    try:
        return brentq(residual, 1e-10, 1-1e-10, xtol=1e-12)
    except:
        return np.nan

def mu_attractor_threshold(Y):
    """Approximate μ on the near-extremal boundary [Santiago Eq. 30]."""
    Y = np.clip(Y, 1e-12, 1.0-1e-12)
    s1mY = np.sqrt(1.0 - Y)
    return z0*np.sqrt(Y) / (s1mY + 1.0)**2

# ── Three-phase integration ───────────────────────────────────────
# Phase 1: μ₀ → μ_h  (mass dissipation, Y rising)
# Phase 2: near-extremal plateau μ_h → z0 at Y≈1  (analytic Eq.32)
# Phase 3: z0 → 0  (down the attractor)
#
# Total physical time = t1 + t2 + t3

mu0 = 0.999
Y0 = 0.5
mu_ext_guess = mu_attractor_threshold(Y0)
regime = "near-extremal / charge-dissipation" if mu0 <= mu_ext_guess else "mass-dissipation"
print(f"\nIC: μ₀={mu0:.3f}  Y₀={Y0:.3f}")
print(f"    M₀={mu0*Ms/M_sun_m:.3e} M☉  Q₀={np.sqrt(Y0)*mu0*Ms/M_sun_m:.3e} M☉")
print(f"    target Q/M={Q_over_M0:.6f}  attractor-scale μ(Y₀)≈{mu_ext_guess:.3e}")
print(f"    regime check: μ₀ {'<' if mu0 < z0 else '>'} z₀ and μ₀ is in the {regime} regime")

if mu0 < 0.1*mu_ext_guess:
    print("WARNING: μ₀ is far below the near-extremal threshold for this Ms normalization.")
    print("         This setup will not climb toward Q/M ≈ 1; it starts deep in the discharge branch.")

# ── Phase 1: integrate until Y → 1-ε ─────────────────────────────
ev_Y1 = lambda t,y: y[1] - (1.0-1e-6)
ev_Y1.terminal = True;  ev_Y1.direction = 1

sol1 = solve_ivp(rhs_tau, [0.0, 1e10], [mu0, Y0],
                 method='Radau', rtol=1e-10, atol=1e-12,
                 max_step=1e-3, events=[ev_Y1], dense_output=True)

tau1_end = sol1.t[-1]
mu_h     = sol1.y[0, -1]   # hanging mass
print(f"\nPhase 1 ends: τ={tau1_end:.4f}  μ_h={mu_h:.4f}  Y={sol1.y[1,-1]:.6f}")

# ── Phase 2: near-extremal plateau  [Eq. 32] ─────────────────────
# Δτ_plateau = [exp((μ-z0)·b0)/b0] evaluated from z0 to μ_h
# = (exp((μ_h-z0)·b0) - 1) / b0
delta_tau2 = (np.exp(np.clip((mu_h-z0)*b0, 0, 500)) - 1.0) / b0
print(f"Phase 2 (plateau): Δτ={delta_tau2:.3e}  Δt={delta_tau2*t_scale_yr:.3e} yr")
tau2_end = tau1_end + delta_tau2

# Build dense representation of plateau:
# μ decreases from μ_h to z0 exponentially slowly, Y stays ≈1
# From Eq.31:  dμ/dτ = -exp((z0-μ)·b0)  →  μ(τ) = z0 + ln(exp((μ_h-z0)·b0) - b0·Δτ)/b0
def mu_plateau(tau):
    """μ(τ) during near-extremal phase, τ measured from phase-1 end."""
    Dtau = tau - tau1_end
    arg  = np.exp((mu_h-z0)*b0) - b0*Dtau
    arg  = np.maximum(arg, 1e-300)
    return z0 + np.log(arg)/b0

N_plateau = 500
tau_p  = np.linspace(tau1_end, tau2_end, N_plateau)
mu_p   = np.array([mu_plateau(t) for t in tau_p])
Y_p    = np.ones(N_plateau)*(1.0-1e-6)

# ── Phase 3: attractor  (z0 → 0) ─────────────────────────────────
ev_mu0 = lambda t,y: y[0] - 1e-4
ev_mu0.terminal = True;  ev_mu0.direction = -1

sol3 = solve_ivp(rhs_tau, [tau2_end, tau2_end+10.0],
                 [z0-1e-6, 1.0-1e-6],
                 method='Radau', rtol=1e-10, atol=1e-12,
                 max_step=5e-4, events=[ev_mu0])

print(f"Phase 3 ends: τ={sol3.t[-1]:.4f}  μ={sol3.y[0,-1]:.4f}")

# ── Stitch all phases ─────────────────────────────────────────────
tau_all = np.concatenate([sol1.t, tau_p, sol3.t])
mu_all  = np.concatenate([sol1.y[0], mu_p, sol3.y[0]])
Y_all   = np.concatenate([sol1.y[1], Y_p,  sol3.y[1]])
Y_all   = np.clip(Y_all, 0.0, 1.0)

# Physical quantities
t_yr  = tau_all * t_scale_yr          # years
M_Msun = mu_all * Ms / M_sun_m        # M☉
Q_Msun = np.sqrt(Y_all) * M_Msun      # M☉  (Q = √Y · M)

# ── Plot ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=False)

# ── Panel 1: M(t) ────────────────────────────────────────────────
ax = axes[0]
ax.plot(t_yr, M_Msun, 'b-', lw=1.5)
ax.axvline(tau1_end*t_scale_yr,  c='gray',   ls=':', lw=1, label='Phase 1→2')
ax.axvline(tau2_end*t_scale_yr,  c='orange', ls=':', lw=1, label='Phase 2→3')
ax.axhline(z0*Ms/M_sun_m, c='red', ls='--', lw=0.8, alpha=0.6,
           label=rf'$M_{{z_0}}={z0*Ms/M_sun_m:.2e}\,M_\odot$')
ax.set_xscale('symlog', linthresh=1e5)
ax.set_yscale('log')
ax.set_ylabel(r'$M$ [$M_\odot$]', fontsize=12)
ax.set_title(rf'Time evolution  ($\mu_0={mu0}$, $Y_0={Y0}$,  '
             rf'$b_0={b0:.0f}$)', fontsize=10)
ax.legend(fontsize=8); ax.grid(True, ls='--', lw=0.4, alpha=0.4)

# ── Panel 2: Q(t) ────────────────────────────────────────────────
ax = axes[1]
ax.plot(t_yr, Q_Msun, 'g-', lw=1.5)
ax.axvline(tau1_end*t_scale_yr,  c='gray',   ls=':', lw=1)
ax.axvline(tau2_end*t_scale_yr,  c='orange', ls=':', lw=1)
ax.set_xscale('symlog', linthresh=1e5)
ax.set_yscale('log')
ax.set_ylabel(r'$Q$ [$M_\odot$]', fontsize=12)
ax.grid(True, ls='--', lw=0.4, alpha=0.4)

# ── Panel 3: Q/M ratio ───────────────────────────────────────────
ax = axes[2]
QoM = np.sqrt(Y_all)
ax.plot(t_yr, QoM, 'r-', lw=1.5)
ax.axvline(tau1_end*t_scale_yr,  c='gray',   ls=':', lw=1)
ax.axvline(tau2_end*t_scale_yr,  c='orange', ls=':', lw=1)
ax.axhline(1.0, c='gray', ls='--', lw=0.8, alpha=0.6, label='Extremal Q=M')
ax.set_xscale('symlog', linthresh=1e5)
ax.set_ylim(0, 1.05)
ax.set_ylabel(r'$Q/M$', fontsize=12)
ax.set_xlabel('$t$ [yr]', fontsize=12)
ax.legend(fontsize=8); ax.grid(True, ls='--', lw=0.4, alpha=0.4)

plt.tight_layout()
plt.savefig('time_evolution.png', dpi=150, bbox_inches='tight')
plt.show()



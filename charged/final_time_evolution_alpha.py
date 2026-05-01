import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import scipy.interpolate as scp
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────
#  Geometrodynamic units: G = c = 1, unit = metre
# ─────────────────────────────────────────────────────────────────────
hbar     = 2.6e-70        # ħ            [m²]
e_std    = 1.38e-36       # electron charge [m]
M_sun_m  = 1477.0         # 1 M☉         [m]
c_SI     = 2.998e8        # c            [m/s]
g_to_m   = 7.425e-31      # G/c² × 10⁻³ [m g⁻¹]
yr_to_s  = 3.15576e7      # 1 year       [s]

Mpl_m    = np.sqrt(hbar)         # Planck mass ≈ 1.612e-35 m
GeV_to_m = Mpl_m / 1.22e19      # 1 GeV/c² in metres ≈ 1.322e-54 m
Mpl = 1.09e-38 * M_sun_m

# ─────────────────────────────────────────────────────────────────────
#  Dark sector  — mχ = 1 GeV,  eχ = 0.3 eₑ
# ─────────────────────────────────────────────────────────────────────
mchi = 1e10 * GeV_to_m   # dark mass in metres
echi = 1e-3 * e_std       # dark charge in metres

Q0 = hbar * echi / (np.pi * mchi**2)   # Schwinger mass-charge scale [m]
print(f"Q₀ = {Q0:.4e} m  ({Q0/M_sun_m:.3e} M☉)")

# ─────────────────────────────────────────────────────────────────────
#  Dimensionless parameters z₀, b₀  (Santiago+ Eqs. 22a-c)
# ─────────────────────────────────────────────────────────────────────
def compute_params(Ms):
    b0  = Ms / Q0
    arg = echi**4 * Ms**2 / (2.0 * np.pi**3 * mchi**2 * hbar**2)
    z0  = (Q0 / Ms) * np.log(arg) if arg > 1.0 else 0.0
    return z0, b0

# Solve for Ms so that z₀ = 0.52682.
# Scan first to find ALL sign changes, then pick the largest-Ms (physical) root.
_target = 0.52682
_lMs    = np.linspace(-120.0, 60.0, 5000)
_vals   = np.array([compute_params(10**l)[0] - _target for l in _lMs])

# Collect bracketing intervals where the function changes sign and is finite
_brackets = [
    (_lMs[i], _lMs[i + 1])
    for i in range(len(_vals) - 1)
    if np.isfinite(_vals[i]) and np.isfinite(_vals[i + 1])
    and _vals[i] * _vals[i + 1] < 0
]
if not _brackets:
    raise RuntimeError("No root found for z₀ = 0.52682 — check dark particle parameters.")

print(f"Root brackets found at log₁₀(Mₛ): {_brackets}")

# Always pick the largest-Ms bracket: that is the physical attractor solution
# (the small-Ms root is a spurious mathematical branch with b₀ ≫ 1 reversed sign)
_a, _b  = _brackets[-1]
log_Ms  = brentq(lambda lM: compute_params(10**lM)[0] - _target,
                 _a, _b, xtol=1e-14)

Ms = 10**log_Ms
z0, b0 = compute_params(Ms)
print(f"Mₛ = {Ms:.4e} m  ({Ms/M_sun_m:.4e} M☉)")
print(f"z₀ = {z0:.6f}  |  b₀ = {b0:.4f}")

# ─────────────────────────────────────────────────────────────────────
#  Greybody factor α(μ, Y)
#  Falls back to the Schwarzschild value α = 0.2679 if file is absent.
# ─────────────────────────────────────────────────────────────────────
alpha_rough = np.genfromtxt("/home/sarthaka/Downloads/blackhawk_v2.3/scripts/greybody_scripts/greybody_factors/charged/new_alpha_charged.txt")

def inter_func(file):
    some_variable = scp.interp1d(file[:,0], file[:,1], kind='cubic', bounds_error=False, fill_value=0.0)
    return some_variable

def alpha_smooth(mu, Y):
        q = float(np.sqrt(np.clip(Y, 0.0, 1.0)))
        alpha_inter = inter_func(alpha_rough)
        val = alpha_inter(q)
        return val

# ─────────────────────────────────────────────────────────────────────
#  Initial conditions
# ─────────────────────────────────────────────────────────────────────
M0g_m = 9.6*10**(12) * g_to_m  # initial mass in metres
mu0   = M0g_m / Ms      # dimensionless initial mass
Qom0   = 0.01           # initial charge-to-mass ratio
Y0    = Qom0**2            # initial (Q/M)²
print(f"\nInitial conditions: μ₀ = {mu0:.4e}  (M₀ = {M0g_m:.4e} m)  ||  Y₀ = {Y0:.4f}  (Q/M)₀ = {np.sqrt(Y0):.4f})")


M0_m = mu0 * Ms   # convert to metres
mu0  = M0_m / Ms        # dimensionless mass
# Y0   = QoM0**2          # dimensionless charge² = (Q/M)²
# print(f"\nμ₀ = {mu0:.4e}  (regime: {'μ₀ < z₀  ← charge-dissipation dominant' if mu0 < z0 else 'μ₀ > z₀  ← Hawking dominant'})")
# print(f"Y₀ = {Y0:.6f}  (Q/M = {QoM0})")

# ─────────────────────────────────────────────────────────────────────
#  Time-domain ODE:  d(μ)/dτ,  d(Y)/dτ   (Santiago+ Eqs. 23-24)
#
#  Physical time conversion (derived from matching Hawking rate):
#       t [s]  =  τ × Mₛ³ / ( ħ × c )
#
#  This is exact up to the O(1) α-normalisation convention in the paper.
# ─────────────────────────────────────────────────────────────────────
def rhs(tau, state):    
    mu, Y = state
    mass = mu * Ms
    charge = np.sqrt(Y) * mass
    alpha = alpha_smooth(mu, Y)
    if mu <= 0.0:
       return [0.0, 0.0]
    Y = np.clip(Y, 1e-14, 1.0 - 1e-6)

    sqY  = np.sqrt(Y)
    s1mY = np.sqrt(1.0 - Y)

    # H(μ, Y)  [Eq. 20] — Hawking term
    H = alpha / mu**2

    # S(μ, Y)  [Eq. 21] — Schwinger term, clipped to avoid overflow
    #   In mass dissipation zone: exponent ≪ 0  →  S ≈ 0  (correct: charge frozen)
    #   On attractor:             exponent = 0  →  S = 1
    #   In charge dissipation:    exponent ≫ 0  →  S ≫ 1  (fast discharge)
    exp_arg = b0 * (z0 - mu * (s1mY + 1.0)**2 / sqY)
    S = np.exp(np.clip(exp_arg, -500.0, 500.0))

    D = (s1mY + 1.0)**4

    dmu = -(H + S * Y**2 / D)                                # Eq. 23
    dY  =  Y * (16*H*mu**2*(s1mY + 1.0) + 2*Y*H*mu**2*(Y-4*(2+s1mY)) + 2*mu**2*Y*S*(-1.0 + Y - s1mY)*Y) / (mu**3 * D) # Eq. 24
    return [scale * dmu, scale * dY]

# ─────────────────────────────────────────────────────────────────────
#  Terminal events
# ─────────────────────────────────────────────────────────────────────
ev_mu  = lambda t, y: y[0] - 1e-8        # μ → 0  (BH fully evaporated)
ev_mu.terminal = True;  ev_mu.direction = -1

ev_ext = lambda t, y: 1.0 - y[1] - 1e-10  # Y → 1  (extremal limit)
ev_ext.terminal = True; ev_ext.direction = -1

# ─────────────────────────────────────────────────────────────────────
#  τ_max:  2× the pure-Hawking evaporation time as a safe upper bound
# ─────────────────────────────────────────────────────────────────────
alp0      = float(alpha_smooth(mu0, Y0))
scale     = mu0**3 / alp0          # σ = τ / scale  →  full evaporation at σ ≈ 1/3
sigma_max = 1e9            # safe upper bound (×2 of the Hawking evap value)
t_max = sigma_max * scale * Ms**3 / (hbar * c_SI)
print(f"\nscale     = {scale:.3e}")
print(f"sigma_max = {sigma_max:.3f}  (pure Hawking σ_evap ≈ 1/3)")
print(f"t_max     ≈ {t_max:.3e} s")
print(f"          ≈ {sigma_max * scale * Ms**3 / (hbar * c_SI * yr_to_s):.3e} yr")

# ─────────────────────────────────────────────────────────────────────
#  Integrate with Radau (L-stable implicit, ideal for stiff ODEs)
# ─────────────────────────────────────────────────────────────────────
# sol1 = solve_config = solve_ivp(
#     rhs,
#     [0.0, sigma_max * 0.5],  # Phase 1: from initial down to near-extremal boundary
#     [mu0, Y0],
#     method='Radau',
#     rtol=1e-10,
#     atol=1e-13,
#     max_step=1e-6,
#     first_step=1e-12,   # <-- FIX 1: Force solver to capture the initial instant discharge
#     events=[ev_mu, ev_ext],
#     dense_output=True
# )

# sol2 = solve_ivp(
#     rhs, [sol1.t[-1], sigma_max], [sol1.y[0, -1], sol1.y[1, -1]],
#     method='Radau', 
#     rtol=1e-10, atol=1e-12, max_step=1e-6,
#     events=[ev_mu, ev_ext], dense_output=True
# )

# ─────────────────────────────────────────────────────────────────────
#  Progress Bar Class
# ─────────────────────────────────────────────────────────────────────
class MassProgressBar:
    """Tracks the evaporation progress based on the Black Hole's mass (mu)."""
    def __init__(self, mu_initial, desc="BH Evaporation"):
        # Set up tqdm bar. elapsed time is shown, and it goes up to 100%
        self.pbar = tqdm(total=100, desc=desc, bar_format="{l_bar}{bar}| {n:.1f}/{total_fmt}% [{elapsed}]")
        self.mu_initial = mu_initial
        self.current_pct = 0.0

    def update(self, mu):
        # Calculate percentage of mass lost
        pct = min(100.0, max(0.0, (1.0 - mu / self.mu_initial) * 100.0))
        
        # Only update the visual bar if progress has moved >= 0.1% to avoid overhead
        if pct - self.current_pct >= 0.1:
            self.pbar.n = pct
            self.pbar.refresh()
            self.current_pct = pct

    def close(self):
        self.pbar.n = 100.0
        self.pbar.refresh()
        self.pbar.close()

# Initialize the progress bar with the initial mass
pbar = MassProgressBar(mu0)

# Wrap your existing RHS function to update the progress bar on every evaluation
def rhs_with_progress(tau, state):
    mu, Y = state
    pbar.update(mu)  # Update progress based on current mass
    return rhs(tau, state)

# ─────────────────────────────────────────────────────────────────────
#  Integrate with Radau (L-stable implicit, ideal for stiff ODEs)
# ─────────────────────────────────────────────────────────────────────
sol1 = solve_ivp(
    rhs_with_progress,       # <-- CHANGED to the wrapped function
    [0.0, sigma_max * 0.5],  # Phase 1: from initial down to near-extremal boundary
    [mu0, Y0],
    method='Radau',
    rtol=1e-10,
    atol=1e-13,
    max_step=1e2,
    first_step=1e-12,        # Force solver to capture the initial instant discharge
    events=[ev_mu, ev_ext],
    dense_output=True
)

sol2 = solve_ivp(
    rhs_with_progress,       # <-- CHANGED to the wrapped function
    [sol1.t[-1], sigma_max], 
    [sol1.y[0, -1], sol1.y[1, -1]],
    method='Radau', 
    rtol=1e-10, 
    atol=1e-12, 
    max_step=1e2,
    events=[ev_mu, ev_ext], 
    dense_output=True
)

# Close the progress bar once both solvers finish
pbar.close()

sigma_arr = np.concatenate((sol1.t, sol2.t))
mu_arr    = np.concatenate((sol1.y[0], sol2.y[0]))
Y_arr     = np.concatenate((sol1.y[1], sol2.y[1]))

sigma_arr = np.concatenate((sol1.t, sol2.t))
mu_arr    = np.concatenate((sol1.y[0], sol2.y[0]))
Y_arr     = np.concatenate((sol1.y[1], sol2.y[1]))

mu_arr = np.clip(mu_arr, 0.0, None)
Y_arr  = np.clip(Y_arr,  0.0, 1.0)

print(f"\nSolver: {sol1.message} / {sol2.message}")
print(f"Steps taken: {len(sigma_arr)}")



# ─────────────────────────────────────────────────────────────────────
#  Convert to physical units
# ─────────────────────────────────────────────────────────────────────
t_s  = sigma_arr * scale * Ms**3 / (hbar * c_SI)   
t_yr = t_s / yr_to_s

# FIX 2: A log axis cannot plot t=0. We give the first element a tiny 
# non-zero value so the starting condition actually appears on the plot.
t_yr_plot = np.maximum(t_yr, 1e-12)

M_g      = mu_arr  * Ms / g_to_m
QoM_arr  = np.sqrt(Y_arr)

print(f"t_final   = {t_s[-1]:.4e} s")
print(f"M_final   = {M_g[-1]:.4e} g")
print(f"Q/M_final = {QoM_arr[-1]:.6f}")

# ─────────────────────────────────────────────────────────────────────
#  Plot: M(t) and Q/M(t)
# ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
ax1, ax2 = axes

# --- Mass ---
ax1.loglog(t_yr_plot, M_g, color='steelblue', lw=1.8)
ax1.set_ylabel(r'Mass $M$ [g]', fontsize=13)
ax1.set_title(
    rf'$M_0 = 10^{{13}}$ g,  $(Q/M)^2_0 = {Y0}$,  '
    rf'$m_\chi = 10^{{10}}\,$GeV,  $e_\chi = {echi/e_std}\,e$' + '\n'
    rf'$M_s = {Ms:.3e}$ m,  $b_0 = {b0:.2f}$,  '
    rf'$z_0 = {z0:.4f}$,  $\mu_0 = {mu0:.2e}$',
    fontsize=9.5)
ax1.grid(True, ls='--', lw=0.4, alpha=0.5)

# FIX 3: Replaced 't_s' with 't_yr_plot' so the axis isn't heavily cropped
if len(t_yr_plot) > 1:
    ax1.set_xlim(left=max(t_yr_plot[1], t_yr_plot[-1] * 1e-8))

# --- Charge-to-mass ---
ax2.semilogx(t_yr_plot, QoM_arr, color='firebrick', lw=1.8)
ax2.axhline(1.0, c='gray', ls='--', lw=1.0, label=r'Extremal $Q = M$')
ax2.set_ylabel(r'$Q/M$', fontsize=13)
ax2.set_xlabel(r'Time $t$ [yrs]', fontsize=13)
ax2.set_xlim(1e-5, t_max / yr_to_s * 1.1)
ax2.set_ylim(0.0, 1.05)
ax2.legend(fontsize=10)
ax2.grid(True, ls='--', lw=0.4, alpha=0.5)

plt.tight_layout()
plt.show()
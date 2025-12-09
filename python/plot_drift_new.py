import numpy as np
import os
import matplotlib.pyplot as plt
from helper_plots import set_size
import librosa
from scipy.io.wavfile import write

from fd_string_model import FD_string_model, get_etas_from_decays, get_T_and_l0_from_f0_beta
from sav_solver import SAVSolver
from results_storage import STATE_STORAGE_CONFIG, DEFAULT_STORAGE_CONFIG
from plotter import NO_PLOTTER_CONFIG

# Output folder
result_folder = "results/drift"

d = os.path.dirname(os.path.abspath(result_folder))
if d and not os.path.exists(d):
    os.makedirs(d, exist_ok=True)

"""
Generates figures of section "to drift or not to drift".
"""

#%%
# System description

# Physical parameters
StringParams = {
    "Ra": 0.597e-3,
    "rho": 7850,
    "E": 2.1e11
}
# Perceptive parameters
f0 = 82.4
beta = 5e-3
T_60_0 = 4
T_60_1000 = 3

# Deduce missing physical parameters
StringParams["T"], StringParams["l0"] = get_T_and_l0_from_f0_beta(f0, beta, StringParams)
StringParams["eta_0"], StringParams["eta_1"] = get_etas_from_decays(T_60_0, T_60_1000, StringParams)

print(StringParams)


model = FD_string_model(44100, **StringParams)

modes = ["KC", "GE", "GE4"] # Nonlinear modes

#%% 
# Simulation parameters
sr = 44100
duration = 10
kappa = 0.9
lambda0s = [0, 1000]
OF = 2 # Over-sampling factor for reference

# Deduce discretization from stability condition
dt = 1 / sr
model.recompute_stability(sr, kappa = kappa)


#%%
# Initial conditions and excitation

# External force (applied at the middle of the string)
def Fext(t):
    Amp = 5
    width = 2e-3
    period = 500 * width
    out = np.zeros(1)
    out[0] = Amp * np.sin(np.pi * t / width) * (t%period < width)
    return out
q0 = np.zeros(model.N)
u0 = np.zeros(model.N)


#%% Run simulations and plot results
fig, axs = plt.subplots(1 + 2*len(lambda0s), 1, figsize=set_size("JAES", height_ratio=0.8), sharex=True)
linestyles = [":", "--", "-."]
for i, lambda0 in enumerate(lambda0s):

  for j, mode in enumerate(modes):
    model.NL_type = mode
    # Compute SAV solution
    solver = SAVSolver(model, sr, lambda0)

    storage = STATE_STORAGE_CONFIG
    storage["Drift"] = True
    storage["q_idx"] = np.array([model.N//2 + 1])
    storage["p_idx"] = None

    solver.integrate(q0, u0, Fext, duration, ConstantRmid=True, plotter_config=NO_PLOTTER_CONFIG, storage_config=storage)
    solver.storage.write(os.path.join(result_folder, f"{mode}/sr{sr}_lambda{lambda0}.h5"))

    f0, _, _ = librosa.pyin(solver.storage.q[:, 0], fmin = 40, fmax = 200, sr = 44100, frame_length = 2048 * 4)
    write(os.path.join(result_folder, f"{mode}/sr{sr}_lambda{lambda0}.wav"), sr, solver.storage.q[:, 0] / np.max(np.abs(solver.storage.q[:, 0])))

    if mode==modes[0] and i==0:
        axs[0].plot(solver.storage.t, [Fext(t) for t in solver.storage.t], color = "black", ls = linestyles[j])
        axs[0].set_ylabel(r"$f_{in}$ [N]")
    axs[2 * i+1].plot(np.linspace(0, duration, len(f0)), f0, label = mode, ls = linestyles[j])
    # Here, we divide espilon by the max observed nonlinear energy to get a relative measure
    print(solver.maxEnl)
    axs[2 * i+2].semilogy(solver.storage.t, np.abs(solver.storage.epsilon / solver.maxEnl), label = mode)

  axs[2*i+1].set_ylabel(r"$f_0$ [Hz]")
  axs[2*i+1].set_ylim(80, 110)
  axs[2*i+2].set_ylabel(r"$\vert\epsilon_{rel}\vert$")
  axs[2*i+2].set_ylim([1e-4, 1.5e3])
  axs[2*i+2].set_yticks([1e-4, 1e-1, 1e2])
  axs[2*i+2].set_yticklabels([1e-4, 1e-1, 1e2])
  axs[2*i+1].text(0.8, 0.6, fr"$\lambda_0 = {lambda0} s^{-1}$", transform = axs[2*i+1].transAxes, color="red", bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
  axs[2*i+2].text(0.8, 0.6, fr"$\lambda_0 = {lambda0} s^{-1}$", transform = axs[2*i+2].transAxes, color="red", bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))

axs[1].legend(loc = "lower center", frameon = True, fancybox = True, bbox_to_anchor = (0.5, 2.1), ncol=3)
axs[4].set_xlim(0, duration)
axs[4].set_ylim(1e-8, 10)
axs[4].set_xlabel(r"Time [s]")
for ax in axs:
    ax.grid()
# Squeeze
fig.tight_layout()
fig.align_ylabels(axs)
fig.subplots_adjust(hspace=0.1, wspace=0.4)
# Save figure
fig.savefig(os.path.join(result_folder, f"test_drift_nl_force.pdf"), bbox_inches='tight')
'''
Run this file to reproduce the figures 2 and 3 from the DAFX25 paper. The only difference between both
is the kappa parameter which tunes the relationship bewtween the spatial timestep and the stability condition.

Important: with the new solver (for JAES), the reference scheme is now a classical Stormer-Verlet scheme. 
In Dafx25, we used an alternative energy-stable schemes for systems with cubic nonlinearity. This reference shceme
and the exact code used for production of the figure in the Dafx paper has been moved in the oldd_solver_dafx25 directory.

Execution of this file show that depsites this difference the reuslts are similar. This proves
that both implementations of the sav solver are equivalent. The new one is intended to comply 
with notations from the JAES paper.
'''
import numpy as np
import h5py
import os
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, NullFormatter

from sav_solver import SAVSolver
from fd_string_model import FD_string_model, DEFAULT_STRING_PARAMS
from plotter import Plotter, DEFAULT_PLOTTER_CONFIG, NO_PLOTTER_CONFIG
from results_storage import ResultsStorage, STATE_STORAGE_CONFIG, DEFAULT_STORAGE_CONFIG
from helper_plots import set_size

#%% Settings: same than in dafx.
# Physical parameters (table 2):
string_params = {
    "l0": 1.1,
    "rho": 8000,
    "T": 60,
    "E": 2e11,
    "eta_0": 0.9,
    "eta_1": 4e-4,
    "Ra": 0.4e-3,
    "NL_type": "GE4"
}

# Exmerimental setting 
A0 = 8e-3 # initial amplitude
duration = 0.1 # This parameter was missing in the Dafx paper, but presented figure indeed corresponds to
# a duration of 0.1s.

# Solver parameters
lambda0s = [0, 1000]
sr0 = 20000
B = 8
OF_ref = 4

kappas = [0.9, 1]

# Storage
result_folder = "results/conv_dafx"

#%% Compute and store results
srs = sr0 * np.pow(2, np.arange(B))
sr_ref = srs[-1] * OF_ref
dts = 1 / srs

d = os.path.dirname(os.path.abspath(os.path.join(result_folder, "settings.h5")))
if d and not os.path.exists(d):
    os.makedirs(d, exist_ok=True)

with h5py.File(os.path.join(result_folder, "settings.h5"), "w") as f:
    f.attrs["sr0"] = json.dumps(sr0)
    f.attrs["srs"] = json.dumps([float(sr) for sr in srs])
    f.attrs["lambda0s"] = json.dumps([float(lambda0) for lambda0 in lambda0s])
    f.attrs["sr_ref"] = json.dumps(float(sr_ref))
    f.attrs["A0"] = json.dumps(A0)

for kappa in kappas:
  for i, sr in enumerate(srs):
    model = FD_string_model(sr0, **string_params)

    model.recompute_stability(sr, kappa = kappa)

    storage = STATE_STORAGE_CONFIG
    storage["q_idx"] = np.array([model.N//2 + 1])
    storage["p_idx"] = None

    def u_func(t):
        return np.zeros(model.Nu)

    # Create the initial condition vector: first mode on the displacement
    # End point are fixed and there are N points in excluding endpoints
    q0 = np.sin(np.pi * np.arange(model.N+2) / (model.N+1))[1:-1] * A0
    u0 = np.zeros(model.N)

    d = os.path.dirname(os.path.abspath(os.path.join(result_folder, f"{kappa}")))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

    for lambda0 in lambda0s:
        solver = SAVSolver(model, float(sr), float(lambda0))
        solver.integrate(q0, u0, u_func, duration, ConstantRmid=True, plotter_config=NO_PLOTTER_CONFIG, storage_config=storage)
        solver.storage.write(os.path.join(result_folder, f"{kappa}/sr{sr}_lambda{lambda0}.h5"))
    print(f"Finished computing for sr = {sr}")

#%% Compute references by oversampling sr_ref_OF times
model = FD_string_model(sr0, **string_params)
model.recompute_stability(sr_ref, 0.9)

storage = STATE_STORAGE_CONFIG
storage["q_idx"] = np.array([model.N//2 + 1])
storage["p_idx"] = None

def u_func(t):
    return np.zeros(model.Nu)

# Create the initial condition vector: first mode on the displacement
# End point are fixed and there are N points in excluding endpoints
q0 = np.sin(np.pi * np.arange(model.N+2) / (model.N + 1))[1:-1] * A0
u0 = np.zeros(model.N)

solver = SAVSolver(model, float(sr_ref), float(0))
solver.integrate_verlet(q0, u0, u_func, duration, ConstantRmid=True, plotter_config=NO_PLOTTER_CONFIG, storage_config=storage)
solver.storage.write(os.path.join(result_folder, f"ref.h5"))
print(f"Finished computing ref")


#%% Plot results
for kappa in kappas:
    #%% Retrieve reference data at midpoint
    with h5py.File(os.path.join(result_folder, f"ref.h5"), "r") as f:
        q_ref = f["q"][...]
        model_setting = json.loads(f.attrs['model_dict'])
        q_ref = q_ref[::int(sr_ref / sr0), 0]

    # %% For each file, compute and store the L2 error on the midpoint of the string at sr0
    def compute_error(q, qref):
        return np.linalg.norm(q - qref, 2) / np.linalg.norm(qref, 2)

    errors = np.zeros((len(srs), len(lambda0s)))
    hs = np.zeros(len(srs))

    for i, sr in enumerate(srs):
        for j, lambda0 in enumerate(lambda0s):
            with h5py.File(os.path.join(result_folder, f"{kappa}/sr{int(sr)}_lambda{int(lambda0)}.h5"), "r") as f:
                q = f["q"][...]
                model_setting = json.loads(f.attrs['model_dict'])
                q = q[::int(sr / sr0), 0]
                errors[i, j] = compute_error(q, q_ref)
                
                hs[i] = model_setting["h"]
    
    #%% Plot
    fig = plt.figure(figsize = set_size("DAFx", fraction=0.5, height_ratio=0.8))
    ax = plt.gca()
    ax.set_xscale("log")
    ax.set_yscale("log")
    for i, lambda0 in enumerate(lambda0s):
        ax.plot(hs, errors[:, i], label=r"$\lambda_0 = $" + f"{lambda0}", ls="dotted", marker = '+', markersize = 5)

    ax.legend(frameon = True) #, ncol=3, bbox_to_anchor=(0.5, 1.05), loc = "upper center")
    ax.set_ylim([np.min(errors) * 0.5, np.max(errors) * 2])

    ax.plot([hs[-1], hs[0]], [errors[-1, 0], errors[-1, 0]/ (hs[-1] / hs[0])**2], linestyle = "--", color="gray")

    # Set x-axis ticks to align with data points
    stride = 2
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_tick_params(which='minor', size=0)
    ax.xaxis.set_tick_params(which='minor', width=0) 
    ax.set_xticks(hs[::stride])
    ax.set_xticklabels([f"{x:.2e}" for x in hs[::stride]])
    # Add a secondary x axis with sampling frequency
    def forward(x):
        return x
    def backward(x):
        return x
    secax = plt.gca().secondary_xaxis('top', functions=(forward, backward))
    secax.set_xlabel('sr (Hz)')
    secax.set_xticks(hs[::stride])
    secax.set_xticklabels([f"{x:.2e}" for x in srs[::stride]])
    secax.xaxis.set_minor_formatter(NullFormatter())
    secax.xaxis.set_tick_params(which='minor', size=0)
    secax.xaxis.set_tick_params(which='minor', width=0)
    plt.xlabel("h (m)")
    plt.ylabel("Relative error e")
    plt.tight_layout()
plt.show()
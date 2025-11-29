import numpy as np
import h5py
import os
import json
import matplotlib.pyplot as plt

from sav_solver import SAVSolver
from fd_string_model import FD_string_model, DEFAULT_STRING_PARAMS
from plotter import Plotter, DEFAULT_PLOTTER_CONFIG, NO_PLOTTER_CONFIG
from results_storage import ResultsStorage, STATE_STORAGE_CONFIG, DEFAULT_STORAGE_CONFIG

"""
Runs convergence test for the string model under a certain configuration.
For now, the reference is taken as the oversampled sav_solver with lambda0 = 0.
TODO: either use a standard verlet solver, easy to implement or build the interface
for simulating models using a external library (e.g. Diffrax)
"""
#%% Settings
# Model and experimental configuration
string_params = DEFAULT_STRING_PARAMS
string_params["NL_type"] = "geom"
A0 = 8e-3 # Initial condition amplitude
duration = 0.05

# Solver
sr0 = 44100
N_srs = 6
sr_ref_OF = 8
kappa = 0.9

lambda0s = [0, 1000]

result_folder = "results/convergence2"



#%% Deduce some other parameters and initialize a string model
srs = sr0 * np.pow(2, np.arange(N_srs))
sr_ref = srs[-1] * sr_ref_OF
dts = 1 / srs

model = FD_string_model(sr0, **string_params)

#%% Write a convergence configuration file to simplify analysis
# ensure directory exists
d = os.path.dirname(os.path.abspath(os.path.join(result_folder, "settings.h5")))
if d and not os.path.exists(d):
    os.makedirs(d, exist_ok=True)

with h5py.File(os.path.join(result_folder, "settings.h5"), "w") as f:
    f.attrs["sr0"] = json.dumps(sr0)
    f.attrs["srs"] = json.dumps([float(sr) for sr in srs])
    f.attrs["lambda0s"] = json.dumps([float(lambda0) for lambda0 in lambda0s])
    f.attrs["sr_ref"] = json.dumps(float(sr_ref))

#%% Run the simulations, storing only the midpoint displacement
for i, sr in enumerate(srs):
    model.recompute_stability(sr, kappa)

    storage = STATE_STORAGE_CONFIG
    storage["q_idx"] = np.array([model.N//2 + 1])
    storage["p_idx"] = None

    def u_func(t):
        return np.zeros(model.Nu)

    # Create the initial condition vector: first mode on the displacement
    # End point are fixed and there are N points in excluding endpoints
    q0 = np.sin(np.pi * np.arange(model.N+2) / (model.N+1))[1:-1] * A0
    u0 = np.zeros(model.N)

    for lambda0 in lambda0s:
        solver = SAVSolver(model, float(sr), float(lambda0))
        solver.integrate(q0, u0, u_func, duration, ConstantRmid=True, plotter_config=NO_PLOTTER_CONFIG, storage_config=storage)
        solver.storage.write(os.path.join(result_folder, f"sr{sr}_lambda{lambda0}.h5"))
    print(f"Finished computing for sr = {sr}")

#%% Compute a reference by oversampling sr_ref_OF times
model.recompute_stability(sr_ref, kappa)

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

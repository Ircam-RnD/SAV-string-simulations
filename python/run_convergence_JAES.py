"""
Runs convergence test for the string model under a certain configuration.
The reference is taken as an oversampled Stormer-Verlet simulation.
"""
import numpy as np
import h5py
import os
import json
import matplotlib.pyplot as plt

from sav_solver import SAVSolver
from fd_string_model import FD_string_model, DEFAULT_STRING_PARAMS
from plotter import Plotter, DEFAULT_PLOTTER_CONFIG, NO_PLOTTER_CONFIG
from results_storage import ResultsStorage, STATE_STORAGE_CONFIG, DEFAULT_STORAGE_CONFIG


#%% Settings
# Model and experimental configuration
string_params = DEFAULT_STRING_PARAMS
modes = ["Linear", "KC", "GE4", "GE"]

A0s = [5e-3, 1e-2, 1.5e-2, 2e-2] # Initial condition amplitude
duration = 0.05

# Solver
sr0 = 44100
N_srs = 6
sr_ref_OF = 8
kappa = 0.9

lambda0s = [-1, 1000]

result_folder = "results/conv_full_2"



#%% Deduce some other parameters
srs = sr0 * np.pow(2, np.arange(N_srs))
sr_ref = srs[-1] * sr_ref_OF
dts = 1 / srs


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
    f.attrs["modes"] = json.dumps(modes)
    f.attrs["A0s"] = json.dumps(A0s)
 
#%% Run the simulations, storing only the midpoint displacement
for mode in modes:
    for A0 in A0s: 
        for i, sr in enumerate(srs):
            string_params["NL_type"] = mode

            model = FD_string_model(sr0, **string_params)

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

            d = os.path.dirname(os.path.abspath(os.path.join(result_folder, f"{mode}/{A0}")))
            if d and not os.path.exists(d):
                os.makedirs(d, exist_ok=True)

            for lambda0 in lambda0s:
                solver = SAVSolver(model, float(sr), float(lambda0))
                if lambda0 ==-1: # Use Stormer Verlet
                    solver.integrate_verlet(q0, u0, u_func, duration, ConstantRmid=True, plotter_config=NO_PLOTTER_CONFIG, storage_config=storage)
                    solver.storage.write(os.path.join(result_folder, f"{mode}/{A0}/sr{sr}_lambda{lambda0}.h5"))
                else:
                    solver.integrate(q0, u0, u_func, duration, ConstantRmid=True, plotter_config=NO_PLOTTER_CONFIG, storage_config=storage)
                    solver.storage.write(os.path.join(result_folder, f"{mode}/{A0}/sr{sr}_lambda{lambda0}.h5"))
                with h5py.File(os.path.join(result_folder, f"{mode}/{A0}/sr{sr}_lambda{lambda0}.h5"), "a") as f:
                    f.attrs["Enlratio"] = json.dumps(solver.Ep_nl(q0) / (solver.Ep_nl(q0) + solver.Ep_lin(q0)))
            print(f"Finished computing for sr = {sr}")

#%% Compute references by oversampling sr_ref_OF times
for mode in modes:
    for A0 in A0s:
        string_params["NL_type"] = mode

        model = FD_string_model(sr0, **string_params)
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
        solver.storage.write(os.path.join(result_folder, f"{mode}/{A0}/ref.h5"))
        print(f"Finished computing ref")

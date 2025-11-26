import numpy as np
import matplotlib.pyplot as plt
from os.path import join

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
A0 = 1e-2 # Initial condition amplitude
duration = 0.1

# Solver
sr0 = 10000
N_srs = 4
kappa = 0.9

lambda0s = [0, 1000]

result_folder = "results/convergence1"



#%% Deduce some other parameters and initialize a string model
srs = sr0 * np.pow(2, np.arange(N_srs + 1))
dts = 1 / srs

model = FD_string_model(sr0, **string_params)

#%% Run the simulations
for i, sr in enumerate(srs):
    model.recompute_stability(sr, kappa)

    def u_func(t):
        return np.zeros(model.Nu)

    # Create the initial condition vector: first mode on the displacement
    # End point are fixed and there are N points in excluding endpoints
    q0 = np.sin(np.pi * np.arange(model.N+2) / model.N)[1:-1] * A0
    u0 = np.zeros(model.N)

    for lambda0 in lambda0s:
        solver = SAVSolver(model, float(sr), float(lambda0))
        solver.integrate(q0, u0, u_func, duration, ConstantRmid=True, plotter_config=NO_PLOTTER_CONFIG, storage_config=STATE_STORAGE_CONFIG)
        solver.storage.write(join(result_folder, f"sr{sr}_lambda{lambda0}.h5"))

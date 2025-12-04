from python.old_solver_dafx25.StringGeom import *
from os import mkdir
from os.path import exists, join
import json
import shutil

# Read config file
with open("config.json") as f:
    config = json.load(f)
print("config : ", config)
 
# String with default parameters
string = StringGeom(**config["string_params"])

sr = config["basesr"]
h, N = string.h_stability(sr, odd=False, alpha = config["alphaCFL"])
print(string.__dict__)

# Initial condition: q0 is set to zero for ease of initialisation of the second order scheme formulation
def qinit(N, h):
    return np.zeros_like(string.hann_init(string.l0/2, 0.2, 4e-3, h, N-1))
def uinit(N, h):
    if config["init"]["mode"] == "modal":
        return (config["init"]["amplitude"] * np.sin(np.pi * np.arange(N+1) / (N)))[1:-1]
    else:
        return string.hann_init(string.l0*config["init"]["position"], config["init"]["width"], config["init"]["amplitude"], h, N-1)

plt.figure()
x = np.linspace(0, string.l0, N+1)[1:-1]
plt.plot(x,uinit(N, 1))
plt.title("Initial velocity")
plt.xlabel("Position (m)")
plt.ylabel("Velocity (m/s)")
plt.show(block=False)

if np.allclose(qinit(100, h), qinit(100, h)[::-1], atol=1e-16) and np.allclose(uinit(100, h), uinit(100, h)[::-1], atol=1e-16):
    print("Initial conditions are symmetric")

# Run simulations
if not exists(config["resultfolder"]):
    mkdir(config["resultfolder"])
shutil.copy("config.json", join(config["resultfolder"], "config.json"))

qmid = np.zeros((len(config["lambda0s"]), int(config["duration"]*sr)))
epsilons = np.zeros((len(config["lambda0s"]), int(config["duration"]*sr)))
for i, lambda0 in enumerate(config["lambda0s"]):
    qsav, psi, epsilons[i] = string.compute_SAV(sr, h, N, qinit(N, h), uinit(N, h), config["duration"], lambda0=lambda0)
    qmid[i] = qsav[:, int(N/2)]
    outpoint = 0.3
    outsigsav = (qsav[1:, int(outpoint*string.l0/h)] - qsav[:-1, int(outpoint*string.l0/h)]) * sr

t = np.arange(int(config["duration"]*sr))/sr

fig, axs = plt.subplots(nrows=1, ncols = 2, figsize=(10, 5))
axs[0].plot(t, qmid.T, label=config["lambda0s"])
axs[0].set_xlabel("Time (s)")
axs[0].set_ylabel("Midpoint displacement (m)")
axs[1].plot(t, epsilons.T, label=config["lambda0s"])
axs[1].set_xlabel("Time (s)")
axs[1].set_ylabel(r"$\epsilon^n$")
plt.legend()
plt.show(block=True)
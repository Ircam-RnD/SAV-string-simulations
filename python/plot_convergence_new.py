import h5py
from os.path import join
from matplotlib.ticker import ScalarFormatter, NullFormatter
import json
import matplotlib.pyplot as plt
import numpy as np
from helper_plots import set_size

#%% Settings
result_folder = "results/conv_full_2"

with h5py.File(join(result_folder, "settings.h5"), "r") as f:
    sr0 = json.loads(f.attrs["sr0"])
    srs = json.loads(f.attrs["srs"])
    lambda0s = json.loads(f.attrs["lambda0s"])
    sr_ref = json.loads(f.attrs["sr_ref"])
    modes = json.loads(f.attrs["modes"])
    A0s = json.loads(f.attrs["A0s"])

lambda0s = lambda0s[-1:]
print(lambda0s)
#%% Init plot
Nrows = 2
Ncols = 2

fig, axs = plt.subplots(Nrows, Ncols, sharex=True, sharey=False, figsize = set_size("JAES", fraction=1, height_ratio=0.6))
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
markers = ['*', '+', '.', 'o', '1', 'v']

for k, mode in enumerate(modes):
    for l, A0 in enumerate(A0s):
        #%% Retrieve reference data at midpoint
        with h5py.File(join(result_folder, f"{mode}/{A0}/ref.h5"), "r") as f:
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
                with h5py.File(join(result_folder, f"{mode}/{A0}/sr{int(sr)}_lambda{int(lambda0)}.h5"), "r") as f:
                    print(f"{mode}/{A0}/sr{int(sr)}_lambda{int(lambda0)}.h5")
                    if f.attrs['success']:
                        model_setting = json.loads(f.attrs['model_dict'])
                        hs[i] = model_setting["h"]
                        q = f["q"][...]
                        q = q[::int(sr / sr0), 0]
                        errors[i, j] = compute_error(q, q_ref)
                    else:
                        errors[i, j] = np.nan()
        
        #%% Plot
        ax = axs[k//2, k%2]
        ax.set_xscale("log")
        ax.set_yscale("log")
        for i, lambda0 in enumerate(lambda0s):
            ax.plot(hs, errors[:, i], label=r"$U_0 = $" + f"{A0} m", ls="dotted", marker = markers[i], markersize = 10, color = colors[l])
        if (k == 3):
            # legend
            ax.legend(bbox_to_anchor=(1, 0.5), loc = "lower left", frameon = True)

            # ax.legend(frameon = True) #, ncol=3, bbox_to_anchor=(0.5, 1.05), loc = "upper center")
        # ax.set_ylim([np.min(errors) * 0.5, np.max(errors) * 2])

        ax.plot([hs[-1], hs[0]], [errors[-1, 0], errors[-1, 0]/ (hs[-1] / hs[0])**2], linestyle = "--", color="gray")

    ax.text(0.1, 0.8, mode, transform = ax.transAxes)
    # Set x-axis ticks to align with data points
    stride = 2
    if (k//2 == 1):
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.xaxis.set_tick_params(which='minor', size=0)
        ax.xaxis.set_tick_params(which='minor', width=0) 
        ax.set_xticks(hs[::stride])
        ax.set_xticklabels([f"{x:.2e}" for x in hs[::stride]])
    else:
        # Add a secondary x axis with sampling frequency
        def forward(x):
            return x
        def backward(x):
            return x
        secax = ax.secondary_xaxis('top', functions=(forward, backward))
        # secax.set_xlabel('sr (Hz)')
        secax.set_xticks(hs[::stride])
        secax.set_xticklabels([f"{x:.2e}" for x in srs[::stride]])
        secax.xaxis.set_minor_formatter(NullFormatter())
        secax.xaxis.set_tick_params(which='minor', size=0)
        secax.xaxis.set_tick_params(which='minor', width=0)
    # plt.xlabel("h (m)")
    # plt.ylabel("Relative error e")
fig.text(0.47, 0.01, 'h (m)', ha='center')
fig.text(0.47, 0.95, 'sr (Hz)', ha='center')
fig.text(0.01, 0.5, 'Relative error e', va='center', rotation='vertical')
plt.tight_layout()
fig.subplots_adjust(left = 0.1, hspace=0.05, wspace=0.2)

plt.show(block = True)
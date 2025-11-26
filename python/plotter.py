import numpy as np
import matplotlib.pyplot as plt

DEFAULT_PLOTTER_CONFIG = {
    "Energy" : True,
    "Power" :True,
    "Drift" : True,
    "q_idx": np.array([0]),
    "p_idx": np.array([0]),
    "SAV": True,
    "interactive": True,
    "stride": 10,
    "delay": 500
}

NO_PLOTTER_CONFIG = {
    "Energy" : False,
    "Power" :False,
    "Drift" : False,
    "q_idx": None,
    "p_idx": None,
    "SAV": False,
    "interactive": False,
    "stride": 10,
    "delay": 500
}

class Plotter():
    """General helper class to plot data from SAVSolver simulations.
    """
    def __init__(self, **kwargs):
        self.__dict__.update(DEFAULT_PLOTTER_CONFIG)
        self.__dict__.update(kwargs)

        # Turns on interaative matplotlib mode if asked
        if self.interactive:
            plt.ion()

    def init_plots(self):
        figsize = (8, 5)
        if self.Energy:
            self.fig_energy = plt.figure(figsize=figsize)
            self.ax_energy = plt.gca()
            self.Etot_l, = self.ax_energy.plot([0], [0], label= "E_tot")
            self.E_k_l, = self.ax_energy.plot([0], [0], label= "E_k")
            self.E_plin_l, = self.ax_energy.plot([0], [0], label= "E_plin")
            self.E_pnl_l, = self.ax_energy.plot([0], [0], label= "E_pnl")
            self.ax_energy.legend(loc= 'best')
            self.fig_energy.suptitle("Energy")
            self.ax_energy.set_ylabel("Energy (J)")
            self.ax_energy.set_xlabel("Time (s)")

        if self.Power:
            self.fig_power, self.axs_power = plt.subplots(2, 1, figsize=figsize, sharex = True)
            self.Pstored_l, = self.axs_power[0].plot([0], [0], label= "P_stored")
            self.Pdiss_l, = self.axs_power[0].plot([0], [0], label= "P_diss")
            self.Pext_l, = self.axs_power[0].plot([0], [0], label= "P_ext")
            self.Ptot_l, = self.axs_power[1].plot([0], [0], linewidth = 0, marker = "x")
            self.axs_power[0].legend(loc= 'best')
            self.fig_power.suptitle("Power")
            self.axs_power[0].set_ylabel("Power (W)")
            self.axs_power[1].set_ylabel("Error on energy balance (J)")
            self.axs_power[1].set_xlabel("Time (s)")
            self.axs_power[1].grid()

        if self.Drift:
            self.fig_drift = plt.figure(figsize=figsize)
            self.ax_drift = plt.gca()
            self.epsilon_l, = self.ax_drift.plot([0], [0])
            self.fig_drift.suptitle("Drift")
            self.ax_drift.set_ylabel("Epsilon")
            self.ax_drift.set_xlabel("Time [s]")

        self.plotq = not (self.q_idx is None)
        self.plotp = not (self.p_idx is None)
        self.plotr = (self.SAV)
        if self.plotq or self.plotp or self.plotr:
            self.Nrows_qpr = int(self.plotq + self.plotp + self.plotr)
            self.fig_qpr, self.axs_qpr = plt.subplots(self.Nrows_qpr, 1, sharex = True, figsize = figsize)
            self.fig_qpr.suptitle("State variables")
            idx = 0
            if self.plotq:
                try:
                    self.ax_q = self.axs_qpr[idx]
                except:
                    self.ax_q = self.axs_qpr
                self.q_l = self.ax_q.plot(np.zeros((1, len(self.q_idx))), np.zeros((1, len(self.q_idx))), label = [f"q_{i}" for i in self.q_idx])
                self.ax_q.legend(loc="upper right")
                self.ax_q.set_xlabel("Time [s]")
                self.ax_q.set_ylabel("q")
                idx+=1
            if self.plotp:
                try:
                    self.ax_p = self.axs_qpr[idx]
                except:
                    self.ax_p = self.axs_qpr
                self.p_l = self.ax_p.plot(np.zeros((1, len(self.p_idx))), np.zeros((1, len(self.p_idx))), label = [f"p_{i}" for i in self.p_idx])
                self.ax_p.legend(loc="upper right")
                self.ax_p.set_xlabel("Time [s]")
                self.ax_p.set_ylabel("p")
                idx+=1
            if self.plotr:
                try:
                    self.ax_r = self.axs_qpr[idx]
                except:
                    self.ax_r = self.axs_qpr
                self.r_l, = self.ax_r.plot([0], [0])
                self.ax_r.set_xlabel("Time [s]")
                self.ax_r.set_ylabel("r")
            self.fig_qpr.tight_layout()
        plt.show(block = False)


    def update_plots(self, data, block = False):
        if not block and not self.interactive:
            return 0
        if (data.i % self.delay) !=0 and not block:
            return 0
        Nt = data.i
        t = data.t[:data.i:self.stride]
        if len(t) >= 2:
            dt = t[1] - t[0]
        else:
            dt = 1
        if self.Energy:
            self.Etot_l.set_xdata(t)
            self.Etot_l.set_ydata(data.Etot[:Nt:self.stride])
            self.E_k_l.set_xdata(t)
            self.E_k_l.set_ydata(data.Ekin[:Nt:self.stride])
            self.E_plin_l.set_xdata(t)
            self.E_plin_l.set_ydata(data.Eplin[:Nt:self.stride])
            self.E_pnl_l.set_xdata(t)
            self.E_pnl_l.set_ydata(data.Epnl[:Nt:self.stride])

            if np.max(data.Etot[:Nt:self.stride]) != 0:
                self.ax_energy.set_ylim(0, 1.2 * np.max(data.Etot[:Nt:self.stride]))
            self.ax_energy.set_xlim(t[0], t[-1])
            self.fig_energy.canvas.draw()
            self.fig_energy.canvas.flush_events()

        if self.Power:
            self.Ptot_l.set_xdata(t)
            self.Ptot_l.set_ydata(data.Ptot[:Nt:self.stride] * dt)
            self.Pstored_l.set_xdata(t)
            self.Pstored_l.set_ydata(data.Pstored[:Nt:self.stride])
            self.Pdiss_l.set_xdata(t)
            self.Pdiss_l.set_ydata(data.Pdiss[:Nt:self.stride])
            self.Pext_l.set_xdata(t)
            self.Pext_l.set_ydata(data.Pext[:Nt:self.stride])

            self.scale_y(self.axs_power[0], data.Pstored[:Nt:self.stride], symmetric=True)
            self.scale_y(self.axs_power[1], data.Ptot[:Nt:self.stride] * dt, symmetric=False)

            self.axs_power[1].set_xlim(t[0], t[-1])
            self.fig_power.canvas.draw()
            self.fig_power.canvas.flush_events()
        
        if self.Drift:
            self.epsilon_l.set_xdata(t)
            self.epsilon_l.set_ydata(data.epsilon[:Nt:self.stride])
            self.scale_y(self.ax_drift, data.epsilon[:Nt:self.stride])
            self.ax_drift.set_xlim(t[0], t[-1])
            self.fig_drift.canvas.draw()
            self.fig_drift.canvas.flush_events()

        if self.plotq:
            for i, q_li in enumerate(self.q_l):
                q_li.set_xdata(t)
                q_li.set_ydata(data.q[:Nt:self.stride, self.q_idx[i]])
            self.scale_y(self.ax_q, data.q[:Nt, self.q_idx])

        if self.plotp:
            for i, p_li in enumerate(self.p_l):
                p_li.set_xdata(t)
                p_li.set_ydata(data.p[:Nt:self.stride, self.q_idx[i]])
            self.scale_y(self.ax_p, data.p[:Nt, self.q_idx])

        if self.plotr:
            self.r_l.set_xdata(t)
            self.r_l.set_ydata(data.r[:Nt:self.stride])
            self.scale_y(self.ax_r, data.r[:Nt:self.stride])

        if self.plotq or self.plotp or self.plotr:
            try:
                for ax in self.axs_qpr:
                    ax.set_xlim((t[0], t[-1]))
            except:
                self.axs_qpr.set_xlim((t[0], t[-1]))
            self.fig_qpr.canvas.draw()
            self.fig_qpr.canvas.flush_events()

        if block:
            plt.show(block=True)
        
    def scale_y(self, ax, data, factor = 1.2, symmetric = False):
        mindata = np.min(data)
        maxdata = np.max(data)
        if symmetric:
            mindata = - max(abs(mindata), abs(maxdata))
            maxdata = max(abs(mindata), abs(maxdata))
        if mindata != maxdata:
            ax.set_ylim(-factor * mindata*np.sign(mindata), factor * maxdata*np.sign(maxdata))

    
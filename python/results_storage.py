import numpy as np

DEFAULT_STORAGE_CONFIG = {
    "Energy" : True,
    "Power" :True,
    "Drift" : True,
    "q_idx": np.array([0]),
    "p_idx": np.array([0]),
    "SAV": True,
    "SolverSetting": True,
    "ModelSetting": True
}

STATE_STORAGE_CONFIG = {
    "Energy" : False,
    "Power" :False,
    "Drift" : True,
    "q_idx": -1,
    "p_idx": -1,
    "SAV": True,
    "SolverSetting": True,
    "ModelSetting": True
}

class ResultsStorage():
    """General helper class to store simulation results from SAVSolver simulations.
    """
    def __init__(self, **kwargs):
        self.__dict__.update(DEFAULT_STORAGE_CONFIG)
        self.__dict__.update(kwargs)

    def reserve(self, t, model, solver):
        self.t = t
        self.Nt = len(t)

        if self.SolverSetting:
            self.solver_dict = solver.setting()
        
        if self.ModelSetting:
            self.model_dict = model.setting()
            print(self.model_dict)

        # Force energy to be computed if power is needed
        if self.Power:
            self.Energy = True

        if self.Energy:
            self.Etot = np.zeros(self.Nt)
            self.Ekin = np.zeros(self.Nt)
            self.Eplin = np.zeros(self.Nt)
            self.Epnl = np.zeros(self.Nt)

        if self.Power:
            # Power is computed on the interleaved time grid
            self.t_inter = (self.t[1:] + self.t[:-1]) / 2
            self.Ptot = np.zeros(self.Nt)
            self.Pdiss = np.zeros(self.Nt)
            self.Pext = np.zeros(self.Nt)
            self.Pstored = np.zeros(self.Nt)

        if self.Drift:
            self.epsilon = np.zeros(self.Nt)

        if self.q_idx == -1:
            self.q_idx = np.arange(model.N)
        if self.q_idx is not None:
            self.q = np.zeros((self.Nt, len(self.q_idx)))

        if self.p_idx == -1:
            self.p_idx = np.arange(model.N)
        if self.p_idx is not None:
            self.p = np.zeros((self.Nt, len(self.p_idx)))

        if self.SAV:
            self.r = np.zeros(self.Nt)

    def store(self, q, p, r, epsilon, i, Rmid, G, u, solver):
        if self.Energy:
            self.Ekin[i] = solver.Ek_tilde(p)
            self.Eplin[i] = solver.Ep_lin(q)
            self.Epnl[i] = solver.Ep_nl_sav(r)
            self.Etot[i] = self.Ekin[i] + self.Eplin[i] + self.Epnl[i]

        if self.Drift:
            self.epsilon[i] = epsilon

        if not (self.q_idx is None):
            self.q[i] = q[self.q_idx]

        if not (self.p_idx is None):
            self.p[i] = p[self.p_idx]

        if self.SAV:
            self.r[i] = r

        if self.Power:
            if i> 0:
                self.Pstored[i] = solver.P_stored(self.Etot[i-1], self.Etot[i])
                self.Pdiss[i] = solver.P_diss(self.plast, p, Rmid)
                self.Pext[i] = solver.P_ext(self.plast, p, G, u)
                self.Ptot[i] = self.Pstored[i] + self.Pdiss[i] + self.Pext[i]
            self.plast = p

        # Keep track of the advance for Plotter function
        self.i = i+1

    def write(self, fname):
        # TODO
        pass
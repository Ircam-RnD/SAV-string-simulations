import numpy as np

DEFAULT_STORAGE_CONFIG = {
    "Energy" : True,
    "Drift" : True,
    "q_idx": np.array([0]),
    "p_idx": np.array([0]),
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

    def reserve(self, t):
        self.t = t
        self.Nt = len(t)

        if self.Energy:
            self.Etot = np.zeros(self.Nt)
            self.Ekin = np.zeros(self.Nt)
            self.Eplin = np.zeros(self.Nt)
            self.Epnl = np.zeros(self.Nt)

        if self.Drift:
            self.epsilon = np.zeros(self.Nt)

        if self.q_idx != None:
            self.q = np.zeros((self.Nt, len(self.q_idx)))

        if self.q_idx != None:
            self.p = np.zeros((self.Nt, len(self.p_idx)))

        if self.SAV:
            self.r = np.zeros(self.Nt)

    def store(self, q, p, r, epsilon, i, solver):
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

        self.i = i+1

    def write(self, fname):
        # TODO
        pass
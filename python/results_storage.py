import numpy as np
import os
import json
try:
    import h5py
except Exception:
    h5py = None

DEFAULT_STORAGE_CONFIG = {
    "Energy" : True,
    "Power" :True,
    "Drift" : True,
    "q_idx": -1,
    "p_idx": -1,
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
        """Write stored results to an HDF5 file. Uses h5py; raise informative
        error if h5py is not installed.
        """
        if h5py is None:
            raise RuntimeError("h5py is required to write results. Install with: pip install h5py")

        # ensure directory exists
        d = os.path.dirname(os.path.abspath(fname))
        if d and not os.path.exists(d):
            os.makedirs(d, exist_ok=True)

        with h5py.File(fname, "w") as f:
            # store metadata dictionaries as JSON attributes
            if hasattr(self, 'solver_dict'):
                f.attrs['solver_dict'] = json.dumps(self.solver_dict)
            if hasattr(self, 'model_dict'):
                f.attrs['model_dict'] = json.dumps(self.model_dict)

            # store configuration flags
            conf = {k: v for k, v in self.__dict__.items() if k in (
                'Energy','Power','Drift','SAV','SolverSetting','ModelSetting')}
            f.attrs['storage_config'] = json.dumps(conf)

            # helper to write arrays if present
            def maybe_write(name, obj):
                if hasattr(self, obj):
                    data = getattr(self, obj)
                    # convert scalars to arrays
                    if np.isscalar(data):
                        data = np.array(data)
                    try:
                        f.create_dataset(name, data=data)
                    except Exception:
                        # fallback: stringify and store as attribute
                        f.attrs[name] = json.dumps(data.tolist() if hasattr(data, 'tolist') else str(data))

            maybe_write('t', 't')
            maybe_write('t_inter', 't_inter')
            maybe_write('Etot', 'Etot')
            maybe_write('Ekin', 'Ekin')
            maybe_write('Eplin', 'Eplin')
            maybe_write('Epnl', 'Epnl')
            maybe_write('Ptot', 'Ptot')
            maybe_write('Pdiss', 'Pdiss')
            maybe_write('Pext', 'Pext')
            maybe_write('Pstored', 'Pstored')
            maybe_write('epsilon', 'epsilon')
            maybe_write('q', 'q')
            maybe_write('p', 'p')
            maybe_write('r', 'r')

            # also store q_idx and p_idx
            if hasattr(self, 'q_idx'):
                try:
                    f.create_dataset('q_idx', data=np.asarray(self.q_idx))
                except Exception:
                    f.attrs['q_idx'] = json.dumps(self.q_idx.tolist() if hasattr(self.q_idx, 'tolist') else str(self.q_idx))
            if hasattr(self, 'p_idx'):
                try:
                    f.create_dataset('p_idx', data=np.asarray(self.p_idx))
                except Exception:
                    f.attrs['p_idx'] = json.dumps(self.p_idx.tolist() if hasattr(self.p_idx, 'tolist') else str(self.p_idx))

    @classmethod
    def load(cls, fname):
        """Load a stored ResultsStorage from an HDF5 file created by `write`.
        Returns a ResultsStorage instance populated with arrays and metadata.
        """
        if h5py is None:
            raise RuntimeError("h5py is required to read results. Install with: pip install h5py")
        if not os.path.exists(fname):
            raise FileNotFoundError(fname)

        with h5py.File(fname, 'r') as f:
            # create an empty instance
            inst = cls()

            # load attributes
            if 'solver_dict' in f.attrs:
                inst.solver_dict = json.loads(f.attrs['solver_dict'])
            if 'model_dict' in f.attrs:
                inst.model_dict = json.loads(f.attrs['model_dict'])
            if 'storage_config' in f.attrs:
                try:
                    conf = json.loads(f.attrs['storage_config'])
                    for k, v in conf.items():
                        setattr(inst, k, v)
                except Exception:
                    pass

            # helper to read dataset if present
            def maybe_read(name):
                if name in f:
                    return f[name][...]
                if name in f.attrs:
                    try:
                        return json.loads(f.attrs[name])
                    except Exception:
                        return f.attrs[name]
                return None

            for name in ('t','t_inter','Etot','Ekin','Eplin','Epnl','Ptot','Pdiss','Pext','Pstored','epsilon','q','p','r'):
                val = maybe_read(name)
                if val is not None:
                    setattr(inst, name, val)

            q_idx = maybe_read('q_idx')
            if q_idx is not None:
                inst.q_idx = np.asarray(q_idx)
            p_idx = maybe_read('p_idx')
            if p_idx is not None:
                inst.p_idx = np.asarray(p_idx)

        return inst
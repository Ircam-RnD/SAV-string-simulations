import numpy as np
from model import Model
from sav_solver import SAVSolver
from results_storage import STATE_STORAGE_CONFIG
from plotter import NO_PLOTTER_CONFIG
from scipy.io.wavfile import write

DEFAULT_STRING_PARAMS = {
    "l0": 1.1,
    "Ra": 0.4e-3,
    "rho": 8000,
    "E": 2e11,
    "T": 60,
    "eta_0": 0.9,
    "eta_1": 4e-4,
    "kc": 1e9,
    "alpha": 1.3,
    "qc": - 5e-3,
    "NL_type": "GE4",
    "expos": 0.5
}

def get_T_and_l0_from_f0_beta(f0, beta, params):
    T = np.sqrt(np.pi**4 * params["E"] * f0**2 * params["Ra"]**6 * params["rho"] / (beta * (1+beta)))
    l0 = np.sqrt(np.pi**3 * params["E"] * params["Ra"]**4 / (4 * beta *T))
    return T, l0

def zeta(omega, gamma2, kappa2):
    return (-gamma2 + np.sqrt(gamma2**2 +4 *kappa2*omega**2))/(2*kappa2)

def get_etas_from_decays(T_60_0, T_60_1000, params):
    mu = params["rho"] * np.pi * params["Ra"]**2
    gamma2 = params["T"] / (mu)
    I = np.pi * params["Ra"]**4 / 4
    kappa2 = params["E"] * I / (mu)
    beta_2_0 = zeta(0, gamma2, kappa2)
    beta_2_1000 = zeta(2*np.pi*1000, gamma2, kappa2)
    eta_0 = np.max(3*np.log(10) / (beta_2_1000 - beta_2_0) * (beta_2_1000 / T_60_0 - beta_2_0/T_60_1000), 0)
    eta_1 = np.max(3*np.log(10) / (beta_2_1000 - beta_2_0) * (-1 / T_60_0 + 1/T_60_1000), 0)
    return eta_0, eta_1


class FD_string_model(Model):
    """Finite difference discretization of in-plane 
    transverse string vibrations. The string might be:
    - Linear
    - Kirchoff-Carrier
    - Third order (cubic) Taylor expansion of geometric nonlinearity
    - Contact-type nonlinearity
    """
    def __init__(self, sr = 44100, **kwargs):
        # Copy string parameters first from Default params
        # and replace parameters given as kwargs
        self.params = DEFAULT_STRING_PARAMS
        self.params.update(**kwargs)
        self.__dict__.update(self.params)
        self.fill_intermediary_physical_params()

        # Samplerate, used for choosing the spatial discretization
        # step as the stability condition
        self.sr = sr

        # Get discretization step from stability condition
        self.h_stability()
        
        # Initialize some storage for d2xq and d4xq
        self.dxq = np.zeros(self.N + 1)
        self.d2xq = np.zeros(self.N)
        self.d4xq = np.zeros(self.N)
        # Compute matrices
        self.build_matrices()

        # One input 
        self.Nu = 1

        # Some constant coefficient multipliers
        self.EnlGeomConst = (self.E * self.A - self.T)/8 * self.h
        self.EnlKCConst = (self.E * self.A)/(8*self.l0) * self.h**2

    def recompute_stability(self, sr, kappa = 0.9):
        self.sr = sr
        # Get discretization step from stability condition
        self.h_stability(kappa = kappa)
        
        # Initialize some storage for d2xq and d4xq
        self.dxq = np.zeros(self.N + 1)
        self.d2xq = np.zeros(self.N)
        self.d4xq = np.zeros(self.N)
        # Compute matrices
        self.build_matrices()

        # One input 
        self.Nu = 1

        # Some constant coefficient multipliers
        self.EnlGeomConst = (self.E * self.A - self.T)/8 * self.h
        self.EnlKCConst = (self.E * self.A)/(8*self.l0) * self.h**2

    def print_perceptual_params(self):
        print(r"Inharmonicity $\Beta = $" + f"{self.beta}")
        print(r"$f_0 = $" + f"{self.f0}")
        print(r"$T_{60}(0) = $" + f"{self.T60(0)}")
        print(r"$T_{60}(1000) = $" + f"{self.T60(2 * np.pi * 1000)}")
    
    def setting(self):
        return {"Name": self.__class__.__name__, "N": self.N, **self.params, "h": self.h} 

    def fill_intermediary_physical_params(self):
        self.A = np.pi*self.Ra**2
        self.I = np.pi*self.Ra**4/4
        self.rhol = self.rho * self.A

        self.c0 = np.sqrt(self.T / self.rhol)
        self.beta = np.pi**2 * self.E * self.I / (self.T * self.l0**2)
        self.f0 = 1 / (2 * self.l0) * self.c0 * np.sqrt(1 + self.beta)
        self.kappa = np.sqrt(self.E * self.I / self.rhol)

    def xi(self, omega):
        return (- self.c0**2 + np.sqrt(self.c0**4 + 4 * self.kappa**2 * omega**2)) / (2 * self.kappa**2)
    def eta(self, omega):
        return self.eta_0 + self.eta_1 * self.xi(omega)
    def T60(self, omega):
        return 6.9 / self.eta(omega)
        
    def h_stability(self, odd = True, kappa = 0.9):
        dt = 1/self.sr
        gamma = dt**2 * self.T + 4*  dt * self.rhol * self.eta_1
        self.h = np.sqrt((gamma + np.sqrt(gamma**2 + 16 * self.rhol * self.E * self.I * dt**2))/ (2 * self.rhol))
        self.N = int(np.floor(kappa * self.l0 / (self.h))) - 1
        if odd:
            if self.N%2 == 0:
                self.N-=1
        else:
            if self.N%2 != 0:
                self.N-=1
        self.h = self.l0 / (self.N + 1)

    def build_matrices(self):
        self.J0 = 1
        self.M = 1 * self.rhol * self.h

    def Rmid(self, q):
        return 2 * self.rhol * self.eta_0 * self.h
    
    def d2xq_op(self, q):
        self.d2xq = -2 *q
        self.d2xq[:-1] += q[1:]
        self.d2xq[1:] += q[:-1]
        self.d2xq /= self.h**2
        return self.d2xq
    
    def d4xq_op(self, d2xq):
        self.d4xq = -2 * d2xq
        self.d4xq[:-1] += d2xq[1:]
        self.d4xq[1:] += d2xq[:-1]
        self.d4xq /= self.h**2
        return self.d4xq

    def K_op(self, q):
        self.d2xq = self.d2xq_op(q)
        self.d4xq = self.d4xq_op(self.d2xq)
        return - self.h * self.T * self.d2xq + self.h * self.E * self.I * self.d4xq
    
    def Rsv_op(self, v):
        # Well in fact its dxp in this case
        self.d2xq = self.d2xq_op(v)
        return - 2 * self.rhol * self.h * self.eta_1 * self.d2xq

    def G(self, q):
        G_val = np.zeros((self.N, self.Nu))
        G_val[self.N//2, 0] = 1
        return G_val

    def Enl(self, q):
        match self.NL_type:
            case "GE4":
                self.dxq[-1] = 0
                self.dxq[:-1] = q
                self.dxq[1:] -= q
                self.dxq /= self.h 
                return self.EnlGeomConst * np.sum((self.dxq)**4)
            case "GE":
                self.dxq[-1] = 0
                self.dxq[:-1] = q
                self.dxq[1:] -= q
                self.dxq /= self.h 
                inter0 = np.sqrt(1 + self.dxq * self.dxq)
                return (self.E * self.A - self.T) * self.h / 2 * (inter0 - 1).dot(inter0 -1)
            case "KC":
                self.dxq[-1] = 0
                self.dxq[:-1] = q
                self.dxq[1:] -= q
                self.dxq /= self.h 
                return self.EnlKCConst * (self.dxq.dot(self.dxq))**2
            case _: # Default to linear
                return 0

    def Fnl(self, q):
        match self.NL_type:
            case "GE4":
                self.dxq[-1] = 0
                self.dxq[:-1] = q
                self.dxq[1:] -= q
                self.dxq /= self.h 
                dxq3 = self.dxq**3
                Dmindxq3 = -1/self.h * (dxq3[1:] - dxq3[:-1])
                return 4 * self.EnlGeomConst * Dmindxq3
            case "GE":
                self.dxq[-1] = 0
                self.dxq[:-1] = q
                self.dxq[1:] -= q
                self.dxq /= self.h 
                inter0 = np.sqrt(1 + self.dxq * self.dxq)
                inter1 = (self.E * self.A - self.T) * (self.dxq * (inter0- 1) / inter0)
                return -1 * (inter1[1:] - inter1[:-1])
            case "KC":
                self.dxq[-1] = 0
                self.dxq[:-1] = q
                self.dxq[1:] -= q
                self.dxq /= self.h 
                return 4 * self.EnlKCConst * (self.dxq.dot(self.dxq)) * (-1/self.h *(self.dxq[1:] - self.dxq[:-1]))
            case _: # Default to linear
                return np.zeros(self.N)

    def EandFnl(self, q):
        match self.NL_type:
            case "GE4":
                self.dxq[-1] = 0
                self.dxq[:-1] = q
                self.dxq[1:] -= q
                self.dxq /= self.h
                dxq3 = self.dxq**3
                Dmindxq3 = -1/self.h * (dxq3[1:] - dxq3[:-1])
                return (self.EnlGeomConst * np.sum((self.dxq)**4),
                        4 * self.EnlGeomConst * Dmindxq3)
            case "GE":
                self.dxq[-1] = 0
                self.dxq[:-1] = q
                self.dxq[1:] -= q
                self.dxq /= self.h 
                inter0 = np.sqrt(1 + self.dxq * self.dxq)
                inter1 = (self.E * self.A - self.T) * (self.dxq * (inter0- 1) / inter0)
                return ((self.E * self.A - self.T) * self.h / 2 * (inter0 - 1).dot(inter0 -1),
                        -1 * (inter1[1:] - inter1[:-1]))
            case "KC":
                self.dxq[-1] = 0
                self.dxq[:-1] = q
                self.dxq[1:] -= q
                self.dxq /= self.h 
                return (self.EnlKCConst * (self.dxq.dot(self.dxq))**2,
                        4 * self.EnlKCConst * (self.dxq.dot(self.dxq)) * (-1/self.h *(self.dxq[1:] - self.dxq[:-1])))
            case _:
                return (self.Enl(q), self.Fnl(q))

if __name__ == "__main__":
    sr = 44100
    model = FD_string_model(sr, NL_type = "GE4")
    model.recompute_stability(sr, kappa=1)
    print("N=", model.N)
    model.print_perceptual_params()
    solver = SAVSolver(model, sr = sr, lambda0=1000)
    solver.check_sizes()
    x = np.linspace(0, 1, model.N+2)
    q0 = np.sin(np.pi*x)[1:-1] * 8e-3
    u0 = np.zeros(model.N)
    def u_func(t):
        return np.zeros(model.Nu)
    solver.integrate(q0, u0, u_func, duration = 1, ConstantRmid=True,
                     plotter_config={"q_idx": np.array([model.N//2]), "p_idx": np.array([model.N//2]), "delay": 5000}, storage_config={})
    solver.storage.write("results/test_string.h5")

    qmid = solver.storage.q[:, model.N//2]
    scaled = np.int16(qmid / np.max(np.abs(qmid)) * 32767)
    write("test_GE4.wav", sr, scaled)
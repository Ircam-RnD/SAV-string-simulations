import numpy as np
from model import Model
from sav_solver import SAVSolver
from results_storage import STATE_STORAGE_CONFIG
from plotter import NO_PLOTTER_CONFIG


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
    "NL_type": "geom"
}

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
        self.__dict__.update(DEFAULT_STRING_PARAMS)
        self.__dict__.update(kwargs)
        self.fill_intermediary_physical_params()

        # Samplerate, used for choosing the spatial discretization
        # step as the stability condition
        self.sr = sr

        # Get discretization step from stability condition
        self.h_stability()
        self.In = np.ones(self.N)
        
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

    def print_perceptual_params(self):
        print(r"Inharmonicity $\Beta = $" + f"{self.beta}")
        print(r"$f_0 = $" + f"{self.f0}")
        print(r"$T_{60}(0) = $" + f"{self.T60(0)}")
        print(r"$T_{60}(1000) = $" + f"{self.T60(2 * np.pi * 1000)}")
        
    def h_stability(self, odd = True, alpha = 0.9):
        dt = 1/self.sr
        gamma = dt**2 * self.T + 4*  dt * self.rhol * self.eta_1
        self.h = np.sqrt((gamma + np.sqrt(gamma**2 + 16 * self.rhol * self.E * self.I * dt**2))/ (2 * self.rhol))
        self.N = int(np.floor(alpha * self.l0 / (self.h)))
        if odd:
            if self.N%2 == 0:
                self.N-=1
        else:
            if self.N%2 != 0:
                self.N-=1
        self.h = self.l0 / (self.N)
        self.In = np.ones(self.N)

    def build_matrices(self):
        self.J0 = 1 / self.h
        self.M = 1 * self.rhol / self.h

    def Rmid(self, q):
        return 2 * self.rhol * self.eta_0 / self.h
    
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
        return - 2 * self.rhol / self.h * self.eta_1 * self.d2xq

    def G(self, q):
        return np.zeros((self.N, self.Nu))

    def Enl(self, q):
        match self.NL_type:
            case "geom":
                self.dxq[-1] = 0
                self.dxq[:-1] = q
                self.dxq[1:] -= q
                self.dxq /= self.h 
                return self.EnlGeomConst * np.sum((self.dxq)**4)
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
            case "geom":
                self.dxq[-1] = 0
                self.dxq[:-1] = q
                self.dxq[1:] -= q
                self.dxq /= self.h 
                dxq3 = self.dxq**3
                Dmindxq3 = -1/self.h * (dxq3[1:] - dxq3[:-1])
                return 4 * self.EnlGeomConst * Dmindxq3
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
            case "geom":
                self.dxq[-1] = 0
                self.dxq[:-1] = q
                self.dxq[1:] -= q
                self.dxq /= self.h
                dxq3 = self.dxq**3
                Dmindxq3 = -1/self.h * (dxq3[1:] - dxq3[:-1])
                return (self.EnlGeomConst * np.sum((self.dxq)**4),
                        4 * self.EnlGeomConst * Dmindxq3)
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
    string_params = {"NL_type": "linear"}
    model = FD_string_model(sr, NL_type = "linear")
    print("N=", model.N)
    model.print_perceptual_params()
    solver = SAVSolver(model, sr = sr)
    solver.check_sizes()
    x = np.linspace(0, 1, model.N+2)
    q0 = np.sin(np.pi*x)[1:-1] * 1
    u0 = np.zeros(model.N)
    def u_func(t):
        return np.zeros(model.Nu)
    solver.integrate(q0, u0, u_func, duration = 1, ConstantRmid=True, plotter_config=NO_PLOTTER_CONFIG, storage_config=STATE_STORAGE_CONFIG)

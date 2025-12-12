import numpy as np
import scipy.sparse as spa
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation as anim
from matplotlib.ticker import ScalarFormatter, NullFormatter
import scipy.linalg as lin
from scipy.io.wavfile import write
from os.path import join
from scipy.signal import resample_poly, decimate
from time import time
from tqdm import tqdm
from helper_plots import set_size
from cycler import cycler
from os.path import exists, join
from helper_plots import *
from librosa import resample

DEFAULT_STRING_PARAMS = {
    "l0": 1.1,
    "Ra": 0.4e-3,
    "rho": 8000,
    "E": 2e11,
    "T": 60,
    "s_0": 0.9,
    "s_1": 4e-4,
    "kc": 1e9,
    "alpha": 1.3,
    "qc": - 5e-3
}

class StringGeom():
    def __init__(self, **kwargs):
        self.__dict__.update(DEFAULT_STRING_PARAMS)
        self.__dict__.update(kwargs)

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
        return self.s_0 + self.s_1 * self.xi(omega)
    def T60(self, omega):
        return 6.9 / self.eta(omega)

    def print_perceptual_params(self):
        print(r"Inharmonicity $\Beta = $" + f"{self.beta}")
        print(r"$f_0 = $" + f"{self.f0}")
        print(r"$T_{60}(0) = $" + f"{self.T60(0)}")
        print(r"$T_{60}(1000) = $" + f"{self.T60(2 * np.pi * 1000)}")
    
    def fTheo(self, N):
        return np.arange(1, N) * self.c0 / (2 * self.l0) * np.sqrt(1 + self.beta * (np.arange(1, N))**2)

    def hann_init(self, center, width, amp, h, N):
        x0 = np.zeros(N)
        for i in range(N-1):
            xcur = (i+1) * h
            dist = np.abs(xcur - center)
            if dist < width:
                x0[i] = amp * 0.5 * (1 + np.cos(np.pi * dist / width))
        return x0
    
    def gauss_init(self, center, width, amp, h, N):
        x0 = np.zeros(N)
        for i in range(N):
            xcur = (i+1) * h
            dist = np.abs(xcur - center)
            x0[i] = amp * np.exp(-dist **2 / (2 * width**2))
        return x0

    def h_stability(self, sr, odd = True, alpha = 1):
        dt = 1/sr
        gamma = dt**2 * self.T + 4*  dt * self.rhol * self.s_1
        h = np.sqrt((gamma + np.sqrt(gamma**2 + 16 * self.rhol * self.E * self.I * dt**2))/ (2 * self.rhol))
        N = int(np.floor(alpha * self.l0 / (h)))
        if odd:
            if N%2 == 0:
                N-=1
        else:
            if N%2 != 0:
                N-=1
        h = self.l0 / (N)
        return h, N
    
    # Nonlinear functions
    def V(self, dxq, q, mode):
        if mode == "geom":
            return np.sum((self.E * self.A - self.T)/8 * self.h * (dxq)**4)
        elif mode == "contact":
            return np.sum(np.maximum( -q + self.qc, 0)**(self.alpha + 1) * self.kc / (self.alpha+1) * self.h)
        elif mode == "KC":
            return (self.E * self.A)/(8*self.l0) * self.h**2 * (dxq.dot(dxq))**2
        else:
            return 0 # Linear string

    def Vprime(self, dxq, q, mode):
        if mode == "geom":
            dxq3 = dxq**3
            Dmindxq3 = -1/self.h * (dxq3[1:] - dxq3[:-1])
            return (self.E * self.A - self.T)/2 * self.h * Dmindxq3
        elif mode == "contact":
            return -np.maximum( -q + self.qc, 0)**(self.alpha) * self.kc * self.h
        elif mode == "KC":
            return (self.E * self.A)/(2*self.l0) * self.h**2 * (dxq.dot(dxq)) * (-1/self.h *(dxq[1:] - dxq[:-1]))
        else:
            return np.zeros_like(q) # Linear string

    def g(self, dxq, q, mode):
        return self.Vprime(dxq, q, mode) / (np.sqrt(2 * self.V(dxq, q, mode)) + 1e-12)

    def compute_SAV(self, sr, h, N, q0, u0, duration, lambda0, Fext = 0, mode="geom", outFnl = False):
        dt = 1/sr
        Ns = int(duration*sr)
        self.h = h

        if not callable(Fext):
            Fext2 = lambda t: 0  # Treat Fext as zero if it's not callable
        else:
            Fext2 = Fext

        # Matrices
        D40 = np.ones(N-1) * 6 / h**4
        D40[0] = 5 / h**4
        D40[-1] = 5 / h**4
        D41 = -4 / h**4
        D42 = 1 / h**4

        dxq = np.zeros(N)

        # Parameters for contact law
        # self.qcV = np.ones(N-1) * self.qc

        # State vectors and initialisation
        q = np.zeros((Ns, N-1))
        q[0] = q0 - dt/2*u0
        q[1] = q0 + dt/2*u0

        psi = np.zeros(Ns)
        dxq[:-1] = q0
        dxq[1:] -= q0
        dxq /= h
        psi[0] = np.sqrt(2 * self.V(dxq, q0, mode))

        epsilons = np.zeros(Ns)

        Fnl = np.zeros_like(q)

        # Maximum value of nonlinear potential for epsilon normalization
        maxPsi = 0

        # Main loop
        for i in tqdm(range(1, Ns-1)):
            # Compute nonlinearity term
            dxq = np.zeros_like(dxq)
            dxq[:-1] = q[i]
            dxq[1:] -= q[i]
            dxq /= h

            gi0 = self.g(dxq, q[i], mode)
            # Modified g
            dxq = np.zeros_like(dxq)
            dxq[:-1] = (q[i] + q[i-1])/2
            dxq[1:] -= (q[i] + q[i-1])/2
            dxq/=h
            Vi = self.V(dxq, (q[i] + q[i-1])/2, mode)
            psii = np.sqrt(2 * Vi)
            epsilon = psi[i-1] - np.sqrt(2 * Vi)
            if (psii > maxPsi):
                maxPsi = psii
            epsilons[i] = epsilon
            #gimod = - lambda0 * epsilon * (q[i] - q[i-1]) /( np.linalg.norm(q[i] - q[i-1], 2) + 1e-12)
            dqmin = 1e-2 * dt
            dq = (q[i] - q[i-1])
            dq = dq * (np.abs(dq) > dqmin)
            gimod = - lambda0 * epsilon * \
                np.sign(dq) * dt / \
                (np.linalg.norm(dq, 1) + 1e-12)
            gi = gi0 + gimod

            # Unrolled matrix products
            righthandterm = (2 * self.rhol * q[i]
                             - ((1-dt*self.s_0) * self.rhol)*q[i-1])
            righthandterm += ((- 1/h**2 * 2 * dt**2 * self.T
                               - dt**2 * self.E * self.I * D40
                               - 4 * self.rhol * self.s_1 * dt / h**2)*q[i]
                              + (4 * self.rhol * dt * self.s_1 / h**2) * q[i-1])
            righthandterm[1:] += (1/h**2 * dt**2 * self.T - dt**2 * self.E * self.I * D41 + 2 * self.rhol *
                                  self.s_1 * dt/h**2) * q[i, :-1] - (2 * self.rhol * self.s_1 * dt/h**2) * q[i-1, :-1]
            righthandterm[:-1] += (1/h**2 * dt**2 * self.T - dt**2 * self.E * self.I * D41 + 2 * self.rhol *
                                   self.s_1 * dt/h**2) * q[i, 1:] - (2 * self.rhol * self.s_1 * dt/h**2) * q[i-1, 1:]
            righthandterm[2:] += (- dt**2 * self.E * self.I * D42) * q[i, :-2]
            righthandterm[:-2] += (- dt**2 * self.E * self.I * D42) * q[i, 2:]
            righthandterm[N//2] += dt**2 / h *Fext2(dt * (i-0.5))
            # Nonlinear part
            righthandterm += dt**2/4 * gi * gi.dot(q[i-1])*1/h
            righthandterm += - 1/h * dt**2 * gi * psi[i-1]

            # Sherman-Morrison
            C1 = dt/2 * gi
            term0 = 1/(self.rhol*(1+dt*self.s_0))
            q[i+1] = term0 * righthandterm \
                - term0**2 * 1/h * C1 * C1.dot(righthandterm) / \
                (1 +term0* 1/h * C1.dot(C1))
            psi[i] = psi[i-1] + 0.5 * gi.dot(q[i+1] - q[i-1])

            Fnl[i] = 1/h * gi * 0.5 * (psi[i-1] + psi[i])
        if outFnl:
            return q, psi, epsilons / maxPsi, Fnl
        return q, psi, epsilons / maxPsi
    
    def compute_SAV_observation(self, sr, h, N, q0, u0, duration, lambda0, Fext = 0, mode="geom"):
        dt = 1/sr
        Ns = int(duration*sr)
        self.h = h

        if not callable(Fext):
            Fext2 = lambda t: 0  # Treat Fext as zero if it's not callable
        else:
            Fext2 = Fext

        # Matrices
        D40 = np.ones(N-1) * 6 / h**4
        D40[0] = 5 / h**4
        D40[-1] = 5 / h**4
        D41 = -4 / h**4
        D42 = 1 / h**4

        dxq = np.zeros(N)

        # Parameters for contact law
        # self.qcV = np.ones(N-1) * self.qc

        # State vectors and initialisation
        qlast = qnow = qnext = np.zeros(N-1)
        qlast = q0 - dt/2*u0
        qnow = q0 + dt/2*u0
        # Storage for the observed point
        Nmid = int(N/2)
        qobs = np.zeros(Ns)
        qobs[0] = qlast[Nmid]
        qobs[1] = qnow[Nmid]
        
        psi = np.zeros(Ns)
        dxq[:-1] = q0
        dxq[1:] -= q0
        dxq /= h
        psi[0] = np.sqrt(2 * self.V(dxq, q0, mode))

        epsilons = np.zeros(Ns)
        # Maximum value of nonlinear potential for epsilon normalization
        maxPsi = 0

        # Main loop
        for i in tqdm(range(1, Ns-1)):
            # Compute nonlinearity term
            dxq = np.zeros_like(dxq)
            dxq[:-1] = qnow
            dxq[1:] -= qnow
            dxq /= h

            gi0 = self.g(dxq, qnow, mode)
            # Modified g
            dxq = np.zeros_like(dxq)
            dxq[:-1] = (qnow + qlast)/2
            dxq[1:] -= (qnow + qlast)/2
            dxq/=h
            Vi = self.V(dxq, (qnow + qlast)/2, mode)
            psii = np.sqrt(2 * Vi)
            epsilon = psi[i-1] - np.sqrt(2 * Vi)
            if (psii > maxPsi):
                maxPsi = psii
            epsilons[i] = epsilon
            #gimod = - lambda0 * epsilon * (q[i] - q[i-1]) /( np.linalg.norm(q[i] - q[i-1], 2) + 1e-12)
            dqmin = 1e-2 * dt
            dq = (qnow - qlast)
            dq = dq * (np.abs(dq) > dqmin)
            gimod = - lambda0 * epsilon * \
                np.sign(dq) * dt / \
                (np.linalg.norm(dq, 1) + 1e-12)
            gi = gi0 + gimod

            # Unrolled matrix products
            righthandterm = (2 * self.rhol * qnow
                             - ((1-dt*self.s_0) * self.rhol)*qlast)
            righthandterm += ((- 1/h**2 * 2 * dt**2 * self.T
                               - dt**2 * self.E * self.I * D40
                               - 4 * self.rhol * self.s_1 * dt / h**2)*qnow
                              + (4 * self.rhol * dt * self.s_1 / h**2) * qlast)
            righthandterm[1:] += (1/h**2 * dt**2 * self.T - dt**2 * self.E * self.I * D41 + 2 * self.rhol *
                                  self.s_1 * dt/h**2) * qnow[:-1] - (2 * self.rhol * self.s_1 * dt/h**2) * qlast[:-1]
            righthandterm[:-1] += (1/h**2 * dt**2 * self.T - dt**2 * self.E * self.I * D41 + 2 * self.rhol *
                                   self.s_1 * dt/h**2) * qnow[1:] - (2 * self.rhol * self.s_1 * dt/h**2) * qlast[1:]
            righthandterm[2:] += (- dt**2 * self.E * self.I * D42) * qnow[:-2]
            righthandterm[:-2] += (- dt**2 * self.E * self.I * D42) * qnow[2:]
            righthandterm[N//2] += dt**2 / h *Fext2(dt * (i-0.5))
            # Nonlinear part
            righthandterm += dt**2/4 * gi * gi.dot(qlast)*1/h
            righthandterm += - 1/h * dt**2 * gi * psi[i-1]

            # Sherman-Morrison
            C1 = dt/2 * gi
            term0 = 1/(self.rhol*(1+dt*self.s_0))
            qnext = term0 * righthandterm \
                - term0**2 * 1/h * C1 * C1.dot(righthandterm) / \
                (1 +term0* 1/h * C1.dot(C1))
            psi[i] = psi[i-1] + 0.5 * gi.dot(qnext - qlast)
            qobs[i+1] = qnext[Nmid]
            qlast = qnow
            qnow = qnext
        return qobs, psi, epsilons / maxPsi

    def compute_semi(self, sr, h, N, q0, u0, duration, Fext = 0):
        dt = 1/sr
        Ns = int(duration*sr)
        self.h = h

        if not callable(Fext):
            Fext2 = lambda t: 0  # Treat Fext as zero if it's not callable
        else:
            Fext2 = Fext

        # Matrices
        Dmin = np.zeros((N, N-1))
        Dmin[:-1, :] = np.diag(np.ones(N-1), 0)
        Dmin[1:, :] -= np.diag(np.ones(N-1), 0)
        Dmin /= h
        Dplus = - Dmin.T
        D2 = Dplus @ Dmin
        D4 = D2 @ D2
        D40 = np.ones(N-1) * 6 / h**4
        D40[0] = 5 / h**4
        D40[-1] = 5 / h**4
        D41 = -4 / h**4
        D42 = 1 / h**4

        baseleftterm = self.rhol * np.eye(N-1) \
            - dt * (-self.rhol * self.s_0 * np.eye(N-1) + self.rhol * self.s_1 * D2)
        base_main = np.diag(baseleftterm, 0)
        base_side = np.diag(baseleftterm, 1)

        # State vectors and initialisation
        q = np.zeros((Ns, N-1))
        q[0] = q0 - dt/2*u0
        q[1] = q0 + dt/2*u0

        dxq = np.zeros(N)
        dxqlast = np.zeros(N)

        dx3 = np.zeros(N)
        dplusdx3 = np.zeros(N-1)

        # Main loop
        for i in tqdm(range(1, Ns-1)):
            # Without matrix product
            righthandterm = (2 * self.rhol * q[i]
                             - ((1-dt*self.s_0) * self.rhol)*q[i-1])
            righthandterm += (- 1/h**2 * 2 * dt**2 *self.T
                               - dt**2 * self.E * self.I * D40) * q[i] \
                              + (2 * self.rhol * dt * self.s_1 / h**2) * q[i-1]
            righthandterm[1:] += (1/h**2 * dt**2 * self.T - dt**2 * self.E * self.I * D41) * q[i, :-1] \
                                - (self.rhol * self.s_1 * dt/h**2) * q[i-1, :-1]
            righthandterm[:-1] += (1/h**2 * dt**2 * self.T - dt**2 * self.E * self.I * D41) * q[i, 1:]\
                                - (self.rhol * self.s_1 * dt/h**2) * q[i-1, 1:]
            righthandterm[2:] += (- dt**2 * self.E * self.I * D42) * q[i, :-2]
            righthandterm[:-2] += (- dt**2 * self.E * self.I * D42) * q[i, 2:]
            # Nonlinearity
            # Compute nonlinearity term
            dxq = np.zeros_like(dxq)
            dxq[:-1] = q[i]
            dxq[1:] -= q[i]
            dxq /= h

            dxqlast = np.zeros_like(dxqlast)
            dxqlast[:-1] = q[i-1]
            dxqlast[1:] -= q[i-1]
            dxqlast /= h

            dx3 = dxq**2 * dxqlast
            dplusdx3 = 1/h*(dx3[1:] - dx3[:-1])

            righthandterm += dt**2 * (self.E * self.A - self.T)/4*dplusdx3
            righthandterm[N//2] += dt**2 / h *Fext2(dt * (i-0.5))
            # Left hand term
            lhs_main = base_main - dt**2/h**2 * (self.E * self.A - self.T)/4 * (-(dxq**2)[:-1] + -(dxq**2)[1:])
            lhs_side = base_side - dt**2/h**2 * (self.E * self.A - self.T)/4 * dxq[1:-1]**2

            q[i+1] = lin.lapack.dgtsv(lhs_side, lhs_main, lhs_side, righthandterm)[3]
        return q
    
    def compute_semi_obs(self, sr, h, N, q0, u0, duration, Fext = 0):
        dt = 1/sr
        Ns = int(duration*sr)
        self.h = h

        if not callable(Fext):
            Fext2 = lambda t: 0  # Treat Fext as zero if it's not callable
        else:
            Fext2 = Fext

        # Matrices
        Dmin = np.zeros((N, N-1))
        Dmin[:-1, :] = np.diag(np.ones(N-1), 0)
        Dmin[1:, :] -= np.diag(np.ones(N-1), 0)
        Dmin /= h
        Dplus = - Dmin.T
        D2 = Dplus @ Dmin
        D4 = D2 @ D2
        D40 = np.ones(N-1) * 6 / h**4
        D40[0] = 5 / h**4
        D40[-1] = 5 / h**4
        D41 = -4 / h**4
        D42 = 1 / h**4

        baseleftterm = self.rhol * np.eye(N-1) \
            - dt * (-self.rhol * self.s_0 * np.eye(N-1) + self.rhol * self.s_1 * D2)
        base_main = np.diag(baseleftterm, 0)
        base_side = np.diag(baseleftterm, 1)

        # State vectors and initialisation
        qlast = q0 - dt/2*u0
        qnow = q0 + dt/2*u0
        qobs = np.zeros(Ns)
        Nmid = int(N/2)
        qobs[0] = qlast[Nmid]
        qobs[1] = qnow[Nmid]

        dxq = np.zeros(N)
        dxqlast = np.zeros(N)

        dx3 = np.zeros(N)
        dplusdx3 = np.zeros(N-1)

        # Main loop
        for i in tqdm(range(1, Ns-1)):
            # Without matrix product
            righthandterm = (2 * self.rhol * qnow
                             - ((1-dt*self.s_0) * self.rhol)*qlast)
            righthandterm += (- 1/h**2 * 2 * dt**2 *self.T
                               - dt**2 * self.E * self.I * D40) * qnow \
                              + (2 * self.rhol * dt * self.s_1 / h**2) * qlast
            righthandterm[1:] += (1/h**2 * dt**2 * self.T - dt**2 * self.E * self.I * D41) * qnow[:-1] \
                                - (self.rhol * self.s_1 * dt/h**2) * qlast[:-1]
            righthandterm[:-1] += (1/h**2 * dt**2 * self.T - dt**2 * self.E * self.I * D41) * qnow[1:]\
                                - (self.rhol * self.s_1 * dt/h**2) * qlast[1:]
            righthandterm[2:] += (- dt**2 * self.E * self.I * D42) * qnow[:-2]
            righthandterm[:-2] += (- dt**2 * self.E * self.I * D42) * qnow[2:]
            # Nonlinearity
            # Compute nonlinearity term
            dxq = np.zeros_like(dxq)
            dxq[:-1] = qnow
            dxq[1:] -= qnow
            dxq /= h

            dxqlast = np.zeros_like(dxqlast)
            dxqlast[:-1] = qlast
            dxqlast[1:] -= qlast
            dxqlast /= h

            dx3 = dxq**2 * dxqlast
            dplusdx3 = 1/h*(dx3[1:] - dx3[:-1])

            righthandterm += dt**2 * (self.E * self.A - self.T)/4*dplusdx3
            righthandterm[N//2] += dt**2 / h *Fext2(dt * (i-0.5))
            # Left hand term
            lhs_main = base_main - dt**2/h**2 * (self.E * self.A - self.T)/4 * (-(dxq**2)[:-1] + -(dxq**2)[1:])
            lhs_side = base_side - dt**2/h**2 * (self.E * self.A - self.T)/4 * dxq[1:-1]**2

            qnext = lin.lapack.dgtsv(lhs_side, lhs_main, lhs_side, righthandterm)[3]
            qobs[i+1] = qnext[Nmid]

            qlast = qnow
            qnow = qnext
        return qobs
    
    def compute_power(self, sr, h, N, q, psi, folder = ""):
        dt = 1/sr
        self.h = h
        
        Dmin = 1/h * spa.diags([1, -1], [0, -1], shape=(N, N-1))
        Dplus = - Dmin.T
        D2 = Dplus @ Dmin

        Mtildeinv = h / self.rhol * (spa.eye(N-1) + (dt**2 / (4*self.rhol) * (self.T * spa.eye(N-1) - self.E * self.I * D2) + dt * self.s_1 * spa.eye(N-1)) @ D2)
        K = h * (-self.T * spa.eye(N-1) + self.E * self.I * D2) @ D2
        R = 2 *self.rhol / h * (self.s_0 * spa.eye(N-1) - self.s_1 * D2)

        Ns = len(q)
        t = np.arange(Ns) * dt
        p = self.rhol / ( dt) * (q[1:] - q[:-1])
        pmid =(p[1:] + p[:-1])/2
        qn = (q[1:] + q[:-1])/2
        Ek = np.zeros(Ns)
        Ep = np.zeros(Ns)
        Enl = np.zeros(Ns)
        E = np.zeros(Ns)
        Pdiss = np.zeros(Ns-1)
        for i in range(1, Ns-1):
            Ep[i] = 0.5 * (qn[i-1].dot(K @ qn[i-1]))
            Ek[i] = 0.5 * (p[i-1].dot(Mtildeinv @ p[i-1]))
            Enl[i] = 0.5 * psi[i-1]**2
            E[i] = Ek[i] + Ep[i] + Enl[i]
            Pdiss[i] = pmid[i-1].dot(R @ pmid[i-1]) * h * h / (self.rhol**2)
        Pstored = (E[1:] - E[:-1]) / dt 
        Ptot = Pstored + Pdiss
        plt.figure(figsize = set_size("DAFx", fraction=0.5))
        plt.plot(t[1:-1], Ek[1:-1], label="Kinetic")
        plt.plot(t[1:-1], Enl[1:-1], label="Nonlinear")
        plt.plot(t[1:-1], Ep[1:-1], label="Potential")
        plt.plot(t[1:-1], E[1:-1], label="Total")
        plt.xlabel("Time (s)")
        plt.ylabel("Energy (J)")
        plt.legend(frameon = True, loc = "upper right")
        plt.ticklabel_format(scilimits=(-2, 2))
        plt.tight_layout()
        plt.savefig(join(folder, "energy.pdf"), bbox_inches="tight")

        plt.figure(figsize = set_size("DAFx", fraction=0.5))
        # plt.plot(t[2:-1], Pdiss[1:-1], label=r"$P_{diss}^{n+\frac{1}{2}}$")
        # plt.plot(t[2:-1], Pstored[1:-1], label=r"$\frac{E^{n+1} - E^n}{dt}$")
        plt.scatter(t[2:-1], Ptot[1:-1], label="Total") #, linewidth=0, marker="x")
        plt.legend(frameon = True, loc = "upper right")
        plt.ticklabel_format(scilimits=(-2, 2))
        plt.xlabel("Time (s)")
        plt.ylabel("Power (W)")
        plt.tight_layout()
        plt.savefig(join(folder, "powers.pdf"), bbox_inches="tight")

        plt.figure(figsize = set_size("DAFx", fraction=0.5, height_ratio=0.7))
        totalDissE = np.cumsum(Pdiss) * dt
        totalDissE[0] = 0
        # plt.plot(t[2:-2], np.cumsum(Ptot[1:-1])[:-1] * dt / E[2])
        plt.plot(t[2:-1], Ptot[1:-1] * dt / E[2])
        plt.grid()
        plt.xlabel("Time (s)")
        plt.ylabel("Energy error")
        plt.tight_layout()
        plt.savefig(join(folder, "energy_balance.pdf"), bbox_inches="tight")
        return Ek, Ep, E, Pdiss, Pstored, Ptot

    def animation_displacement(self, qs, h, N, sr, slow_factor = 100):
        dt = 1/sr
        time = np.linspace(0, dt * len(qs[0]), len(qs[0]))
        space = np.linspace(0, h*N, N+1)
        space = space[1:-1]
        self.h = h

        fig, axs = plt.subplots(1, 1)
        qtrace, = axs.plot(space, qs[0][0], label = "SAV", linestyle="--")
        qtrace2, = axs.plot(space, qs[0][0], label = "Semi-implicit", linestyle=":")
        axs.set_xlabel("Position")
        axs.set_ylabel("Displacement")
        axs.set_ylim([-1.2 * np.max(np.abs(qs[0])), 1.2 * np.max(np.abs(qs[0]))])
        axs.legend()

        def pos(frame):
            qtrace.set_ydata(qs[0][frame])
            qtrace2.set_ydata(qs[1][frame])
            return (qtrace, qtrace2, )
        animation = anim.FuncAnimation(fig=fig,
                                       func=pos,
                                       frames=len(qs[0]),
                                       interval=int(dt * 1000 * slow_factor),
                                       blit=True)
        plt.show()
        return fig

    def run_convergence(self, qinit, uinit, srs, sr_ref, lambda0s, duration, h, N, error="global", folder="", position=0.3, plot = True, mode = "geom"):
        ''' Errors are computed using L2 spatial norm. The sampling frequency is changed but not the spatial discretization (given by N). 
        Results are aligned in time by computing midpoint values (as q is initially on the dual grid t^{n-1/2}, t^{n+1/2}). Secondly,
        they are decimated to srs[0] to compute the error.
        '''
        t0 = time()
        Nmax = int(srs[0] * duration)-1

        def align_q_results(q, sr, sr_target):
            """ Midpoint rule followed by decimation
            """
            q = (q[:-1] + q[1:])/2
            q = q[::int(sr/sr_target), :]
            q = q[:Nmax, :]
            return q

        # Initial conditions
        q0 = qinit(N, h)
        u0 = uinit(N, h)

        # Compute reference solution
        dt_ref = 1/sr_ref
        if mode=="geom":
            qref = self.compute_semi(sr_ref, h, N, q0, u0, duration)
        else:
            qref, _, _ = self.compute_SAV(sr_ref, h, N, q0, u0, duration, lambda0=lambda0s[-1], mode = mode)
        print("Finished computing reference solution")

        qref = align_q_results(qref, sr_ref, srs[0])

        if plot:
            fig = plt.figure()
            ax = plt.gca()
            plt.title("FFT of center displacement")
            fftqref = np.fft.rfft(qref[:, int(N/2)], norm="forward")
            fftscale = np.max(np.abs(fftqref))
            freq = np.fft.rfftfreq(len(qref[:, int(N/2)]), dt_ref)
            ax.plot(freq, 20*np.log10(np.abs(fftqref) / fftscale), label = "reference")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Amplitude")
            ax.legend()
            plt.draw()

        # Compute solutions and errors for each configurations
        errors_sav  = np.zeros((len(lambda0s), len(srs)))
        errors_semi = np.zeros(len(srs))
        for i, sr in enumerate(srs):
            dt = 1/sr
            for j, lambda0 in enumerate(lambda0s):
                qsav, psisav, _ = self.compute_SAV(sr, h, N, q0, u0, duration, lambda0 = lambda0, mode=mode)
                qsav = align_q_results(qsav, sr, srs[0])
                errors_sav[j, i] = np.linalg.norm(qsav - qref, 2) / np.linalg.norm(qref, 2)
                # Audio
                psav = (qsav[1:, int(position*self.l0/h)] - qsav[:-1, int(position*self.l0/h)]) *sr
                psav_resampled = resample_poly(psav, 1, int(sr/np.min(srs))).astype(np.float32)
                write(join(folder, f"semi_{sr}.wav"), np.min(srs), psav_resampled/np.max(np.abs(psav_resampled)))

            if mode=="geom":
                qsemi = self.compute_semi(sr, h, N, q0, u0, duration)
                qsemi = align_q_results(qsemi, sr, srs[0])
                errors_semi[i] = np.linalg.norm(qsemi - qref, 2) / np.linalg.norm(qref, 2)
                # Audio
                psemi = (qsemi[1:, int(position*self.l0/h)] - qsemi[:-1, int(position*self.l0/h)]) *sr
                psemi_resampled = resample_poly(psemi, 1, int(sr/np.min(srs))).astype(np.float32)
                write(join(folder, f"semi_{sr}.wav"), np.min(srs), psemi_resampled/np.max(np.abs(psemi_resampled)))
            print("Finished computing solutions for sr = ", sr, " Hz")

        t1 = time()
        print(f"Elapsed simulation time : ", "%.2f" % (t1-t0), " s")

        # Convergence plot
        fig = plt.figure(figsize=(6, 3))
        # 2nd order slope
        plt.plot([srs[0], srs[-1]], [errors_sav[0, 0], errors_sav[0, 0]/ (srs[-1] / srs[0])**2], linestyle = "--", color="gray", label = "second order slope")
        # Reference algotrithm for the cubic nonlinearity
        if mode=="geom":
            plt.plot(srs, errors_semi, label = "reference algorithm", color="blue", ls="-.", marker= "x", markersize = 15) 
        # SAV algorithm
        cycle = cycler(color = mpl.colormaps["Reds"](np.linspace(0.3, 1, len(lambda0s))))
        plt.gca().set_prop_cycle(cycle)
        for i, lambda0 in enumerate(lambda0s):
            plt.loglog(srs, errors_sav[i], label=r"$\lambda = $" + f"{lambda0}", ls="dotted", marker = "+", markersize = 15)
        # Naming / Style
        plt.xlabel("sr [Hz]")
        plt.ylabel("Relative error e")
        plt.grid()
        plt.legend()
        plt.tight_layout()
        fig.savefig(join(folder, f"conv.pdf"), bbox_inches="tight")
        plt.show()
        return errors_sav, errors_semi
    
    def run_convergence_space_time(self, qinit, uinit, srs, sr_ref, lambda0s, duration, error="global", folder="", position=0.3, alpha = 1, plot=True, mode = "geom"):    
        t0 = time()
        sr0 = srs[0]
        dt0 = 1 / sr0
        duration = int(duration / dt0) * dt0
        # Compute reference solution
        dt_ref = 1/sr_ref
        href, Nref = self.h_stability(sr_ref, odd=False, alpha=alpha)
        if mode=="geom":
            # qref= self.compute_semi(sr_ref, href, Nref, qinit(Nref, href), uinit(Nref, href), duration)
            qref= self.compute_semi_obs(sr_ref, href, Nref, qinit(Nref, href), uinit(Nref, href), duration)
        else:
            # qref, _, _ = self.compute_SAV(sr_ref, href, Nref, qinit(Nref, href), uinit(Nref, href), duration, lambda0=lambda0s[-1], mode=mode)
            qref, _, _ = self.compute_SAV_observation(sr_ref, href, Nref, qinit(Nref, href), uinit(Nref, href), duration, lambda0=lambda0s[-1], mode=mode)
        print("Finished computing reference solution")
        
        pref = np.zeros(len(qref))
        pref[:-1] = (qref[1:] - qref[:-1]) * sr_ref
        # Decimate to sr 0
        # pref = resample(pref, orig_sr = sr_ref, target_sr = sr0)
        pref = decimate(pref, int(sr_ref/sr0), n=35, ftype="fir")
        qref = qref[::int(sr_ref / sr0)]
        # compute velocity

        if plot:
            fig = plt.figure()
            ax = plt.gca()
            plt.title("FFT of center displacement")
            fftqref = np.fft.rfft(qref, norm="forward")
            fftscale = np.max(np.abs(fftqref))
            freq = np.fft.rfftfreq(len(qref), dt0)
            ax.plot(freq, 20*np.log10(np.abs(fftqref)/fftscale), label = "reference")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Amplitude")
            ax.legend()
            plt.draw()

        # Compute solutions for each configurations
        errors_sav  = np.zeros((len(lambda0s), len(srs)))
        errors_semi = np.zeros(len(srs))
        hs = np.zeros(len(srs))
        for i, sr in enumerate(srs):
            print("Computing solutions for sr = ", sr, " Hz")
            dt = 1/sr
            h, N = self.h_stability(sr, odd=False, alpha=alpha)
            hs[i] = h
            for j, lambda0 in enumerate(lambda0s):
                qsav, psisav, _ = self.compute_SAV_observation(sr, h, N, qinit(N, h), uinit(N, h), duration, lambda0 = lambda0, mode=mode)
                psav = np.zeros(len(qsav))
                psav[:-1] = (qsav[1:] - qsav[:-1]) * sr
                # Decimate to sr0
                # psav = resample(psav, orig_sr = sr, target_sr = sr0)
                if sr>sr0:
                    psav = decimate(psav, int(sr/sr0), n=35, ftype="fir")
                qsav = qsav[::int(sr / sr0)]
                # print(np.linalg.norm(qref, 2), np.linalg.norm(qsav, 2))
                if mode == "contact":
                    errors_sav[j, i] = np.linalg.norm(psav[:-1] - pref[:-1], 2) / np.linalg.norm(pref[:-1], 2)
                else:
                    errors_sav[j, i] = np.linalg.norm(qsav - qref, 2) / np.linalg.norm(qref, 2)
                write(join(folder, f"sav_{sr}_{lambda0}.wav"), sr0, psav/np.max(np.abs(psav)))

                if plot:
                    fftq = np.fft.rfft(psav, norm="forward")
                    freq = np.fft.rfftfreq(len(psav), dt)
                    ax.plot(freq, 20*np.log10(np.abs(fftq)/fftscale), label=f"sav {sr} {lambda0}")
                    ax.legend()
                    plt.draw()

            if mode == "geom":
                qsemi = self.compute_semi_obs(sr, h, N, qinit(N, h), uinit(N, h), duration)
                psemi = np.zeros(len(qsemi))
                
                psemi[:-1] = (qsemi[1:] - qsemi[:-1]) * sr
                if sr>sr0:
                    psemi = decimate(psemi, int(sr/sr0), n=35, ftype="fir")
                qsemi = qsemi[::int(sr / sr0)]
                if mode == "contact":
                    errors_semi[i] = np.linalg.norm(psemi[:-1] - pref[:-1], 2) / np.linalg.norm(pref[:-1], 2)
                else:
                    errors_semi[i] = np.linalg.norm(qsemi - qref, 2) / np.linalg.norm(qref, 2)
                write(join(folder, f"semi_{sr}.wav"), sr0, psemi/np.max(np.abs(psemi)))

                if plot:
                    fftq = np.fft.rfft(psemi[:], norm="forward")
                    freq = np.fft.rfftfreq(len(psemi[:]), dt)
                    ax.plot(freq, 20*np.log10(np.abs(fftq)/fftscale), label=f"semi {sr}")
                    ax.legend()
                    plt.xlim([0, 5000])
                    plt.draw()
                    fig.savefig(join(folder, f"ffts.png"))
            

            print("Finished computing solutions for sr = ", sr, " Hz")
        t1 = time()
        print(f"Elapsed simulation time : ", "%.2f" % (t1-t0), " s")
        # fig = plt.figure(figsize = set_size("DAFx", fraction=0.5))
        # #plt.plot([srs[0], srs[-1]], [0.5 * errors_semi[0], 0.5 * errors_semi[0]/ (srs[-1] / srs[0])**2], linestyle = "--", color="gray", label = "Second order slope")
        # plt.plot([srs[0], srs[-1]], [0.5 * errors_semi[0], 0.5 * errors_semi[0]/ (srs[-1] / srs[0])], linestyle = "--", color="gray", label = "First order slope")
        # plt.plot(srs, errors_semi, label = "reference algorithm", color="blue", ls="-.", marker= "x", markersize = 15)
        # plt.xlabel("sr (Hz])")
        # plt.ylabel("Relative error e")
        # plt.grid()
        # #draw_loglog_slope(fig, plt.gca(), origin=(np.min(srs), np.min(errors_semi)), inverted=False, width_inches=1e5, slope=2, color="black")
        # cycle = cycler(color = mpl.colormaps["Reds"](np.linspace(0.3, 1, len(lambda0s))))
        # plt.gca().set_prop_cycle(cycle)
        # for i, lambda0 in enumerate(lambda0s):
        #     plt.loglog(srs, errors_sav[i], label=r"$\lambda_0 = $" + f"{lambda0}", ls="dotted", marker = "+", markersize = 15)
        # plt.legend(frameon = True)
        # plt.tight_layout()
        # fig.savefig(join(folder, f"conv_sr.pdf"), bbox_inches="tight")

        fig = plt.figure(figsize = set_size("DAFx", fraction=0.5, height_ratio=0.8))
        ax = plt.gca()
        ax.set_xscale("log")
        ax.set_yscale("log")
       
        # 2nd order convergence triangle
        # triangle = mpl.patches.Polygon([[hs[-3], 0.5 * errors_semi[0]], [hs[-3], 0.5 * errors_semi[0]/ (hs[-1] / hs[-3])**2], [hs[-1], 0.5 * errors_semi[0]]], closed=True, color="gray", alpha=0.2)
        # ax.add_patch(triangle)
        # Add numbers
        # ax.text(np.sqrt(hs[-1] * hs[-3]), 0.5 * errors_semi[0], "1", fontsize=8, ha='center', va='top')
        # ax.text(hs[-3], np.sqrt(0.5 * errors_semi[0] * 0.5 * errors_semi[0]/ (hs[-1] / hs[-3])**2), "2", fontsize=8, ha='left', va='center')


        ax.plot(hs, errors_semi, label = "reference", color="blue", ls="-.", marker= "x", markersize = 5)
        ax.grid()
        cycle = cycler(color = mpl.colormaps["Reds"](np.linspace(0.3, 1, len(lambda0s))))
        ax.set_prop_cycle(cycle)
        for i, lambda0 in enumerate(lambda0s):
            ax.plot(hs, errors_sav[i], label=r"$\lambda_0 = $" + f"{lambda0}", ls="dotted", marker = "+", markersize = 5)
        ax.legend(frameon = True) #, ncol=3, bbox_to_anchor=(0.5, 1.05), loc = "upper center")
        ax.set_ylim([np.min(errors_semi) * 0.5, np.max(errors_sav) * 2])

        ax.plot([hs[-1], hs[0]], [errors_sav[0, -1], errors_sav[0, -1]/ (hs[-1] / hs[0])**2], linestyle = "--", color="gray")

        # Set x-axis ticks to align with data points
        stride = 2
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.xaxis.set_tick_params(which='minor', size=0)
        ax.xaxis.set_tick_params(which='minor', width=0) 
        ax.set_xticks(hs[::stride])
        ax.set_xticklabels([f"{x:.2e}" for x in hs[::stride]])
        # Add a secondary x axis with sampling frequency
        def forward(x):
            return x
        def backward(x):
            return x
        secax = plt.gca().secondary_xaxis('top', functions=(forward, backward))
        secax.set_xlabel('sr (Hz)')
        secax.set_xticks(hs[::stride])
        secax.set_xticklabels([f"{x:.2e}" for x in srs[::stride]])
        secax.xaxis.set_minor_formatter(NullFormatter())
        secax.xaxis.set_tick_params(which='minor', size=0)
        secax.xaxis.set_tick_params(which='minor', width=0)
        plt.xlabel("h (m)")
        plt.ylabel("Relative error e")
        plt.tight_layout()
        fig.savefig(join(folder, f"conv_hs.pdf"), bbox_inches="tight")
        plt.show()
        return errors_sav, errors_semi

if __name__ == "__main__":
    string = StringGeom()

    def F(t):
        return np.zeros_like(t)
        Amp = 2
        width = 3e-3
        period = 1
        return Amp/5 * np.sin(np.pi * t / width) * (t%period < width) * (t< 3) \
            + Amp * np.sin(np.pi * t / width) * (t%period < width) * (t> 3) * (t<7) \
            + Amp/5 * np.sin(np.pi * t / width) *  (t%period < width)* (t> 7)

    sr = 44100
    dt = 1/sr
    duration = 0.1
    h, N = string.h_stability(sr, odd=False, alpha = 0.8)
    print(N)
    print(string.__dict__)
    q0 = string.gauss_init(string.l0/2, 0.05, 0.004, h, N-1)
    # q0 = np.zeros_like(q0)
    # q0 = np.sin(np.pi / N * (np.arange(N-1) + 1)) * 1e-2
    #u0 = string.hann_init(string.l0/2, 0.2, 1, h, N-1)
    #u0 = np.zeros_like(u0)
    u0 = np.sin(np.pi / N * (np.arange(N-1) + 1)) * 0
    if np.allclose(q0, q0[::-1], atol=1e-16) and np.allclose(u0, u0[::-1], atol=1e-16):
        print("Initial conditions are symmetric")
    qsav, psi, epsilon= string.compute_SAV(sr, h, N, q0, u0, duration, lambda0=10000, Fext=F, mode= "geom")
    plt.figure()
    plt.plot(np.arange(len(epsilon)) * dt, epsilon, label=r"$\epsilon$")
    # # plt.plot(np.arange(len(psi)) * dt, psi, label="SAV")
    # plt.legend()
    # plt.grid()

    # resultfolder = "./results/"
    # if not exists(resultfolder):
    #     mkdir(resultfolder)
    plt.figure()
    print(np.max(np.abs((qsav[1:, int(N/2)] - qsav [:-1, int(N/2)])/dt)))
    plt.plot(qsav[1:, int(N/2)] - qsav [:-1, int(N/2)], label="SAV")

    Ek, Ep, E, Pdiss, Pstored, Ptot = string.compute_power(sr, h, N, qsav, psi, "/Users/risse/work/these/codes manuscrit/SAV_drift")
    qsemi = string.compute_semi(sr, h, N, q0, u0, duration)
    fig1 = string.animation_displacement([qsav, qsemi], h, N, sr, slow_factor=100)
    # outpoint = 0.5
    # outsigsav = (qsav[1:, int(outpoint*string.l0/h)] - qsav[:-1, int(outpoint*string.l0/h)]) *sr
    # outsigsav = qsav[:, int(outpoint*string.l0/h)]
    # # outsigsemi = qsemi[:, int(outpoint*string.l0/h)]
    # #outsigsemi = (qsemi[1:, int(outpoint*string.l0/h)] - qsemi[:-1, int(outpoint*string.l0/h)]) *sr

    # plt.figure()
    # plt.plot(outsigsav, label="sav")
    # # plt.plot(outsigsemi, label="semi")
    # plt.legend()
    # sd.play(outsigsav/np.max(np.abs(outsigsav)), samplerate=sr, blocking=True)
    #sd.play(outsigsemi/np.max(np.abs(outsigsemi)), samplerate=sr, blocking=True)
    # write("sound2.wav", 44100, outsigsav/np.max(np.abs(outsigsav)))
    plt.show(block=True)
import numpy as np
import matplotlib.pyplot as plt
from time import sleep
from model import Model
from results_storage import ResultsStorage, DEFAULT_STORAGE_CONFIG
from plotter import Plotter, DEFAULT_PLOTTER_CONFIG

class SAVSolver():
    """General SAV solver using same notations than in the JAES paper.
    """
    def __init__(self, model, sr = 44100, lambda0 = 0):
        ### System definition ###
        self.model = model

        ### Numerical parameters ###
        # Sampling
        self.sr = sr
        self.dt = 1/self.sr
        self.dt2 = self.dt**2
        # Shifting constant
        self. C0 = 0
        # Control parameter
        self.lambda0 = lambda0
        # Numerical epsilon
        self.Num_eps = 1e-16

        ### Some vector initialization ###
        self.Gn = np.zeros(self.model.N)
        self.Rmidn = np.zeros(self.model.N)
        self.A0_inv_n = np.zeros(self.model.N)
        self.gn = np.zeros(self.model.N)

        self.RHSn = np.zeros(self.model.N)

        ### Intermediary fixed quantities ###
        self.M_J0dt = self.model.M / (self.dt * self.model.J0)

        ### Check dimensions ###
        self.check_sizes()

    def setting(self):
        """Returns a dictionnary with solver settings.
        Returns:
            (dict): solver setings
        """
        return {"sr": self.sr, "dt": self.dt, "num_eps": self.Num_eps, "C0": self.C0, "lambda0": self.lambda0}

    def check_sizes(self):
        """Checks that the underlying model has coherent matrices and operators
        shapes.
        """
        # Check that all matrices and operator provided to describe the 
        # system are of the right dimensions
        x = np.zeros(self.model.N)
        u = np.zeros((self.model.N, self.model.Nu))
        J0x = np.zeros(self.model.N)
        try:
            J0x = self.model.J0 * x
            if (J0x.shape[0] != self.model.N or J0x.ndim != 1):
                raise Exception(f"J0 x must return a vector of length {self.model.N} but has shape {J0x.shape}")
        except:
            raise Exception(f"J0 cannot be applied to a vector of length {self.model.N}")
        
        try:
            Mx = self.model.M * x
            if (Mx.shape != J0x.shape):
                raise Exception(f"M x must return a vector of length {self.model.N} but has shape {Mx.shape}")
        except:
            raise Exception(f"M cannot be applied to a vector of length {self.model.N}")
        
        try:
            Rmidx = self.model.Rmid(x) * x
            if (Rmidx.shape != J0x.shape):
                raise Exception(f"Rmid(x) x must return a vector of length {self.model.N} but has shape {Rmidx.shape}")
        except:
            raise Exception(f"Rmid(x) cannot be computed, or applied to a vector of length {self.model.N}")
        
        try:
            Kq = self.model.K_op(x)
        except:
            raise Exception(f"K_op function not working with input vector of size {self.model.N}")
        if (Kq.shape != J0x.shape):
            raise Exception(f"Kq must have {self.model.N} elements but has shape {Kq.shape}")
        Rsvq = self.model.Rsv_op(x)
        if (Rsvq.shape != J0x.shape):
            raise Exception(f"Rsvq must have {self.model.N} elements but has shape {Rsvq.shape}")
        try: 
            self.Gn = self.model.G(u)
        except:
            Exception(f"G function not working with input vector of shape {self.u.shape}")
        if (self.Gn.shape != (self.model.N, self.model.Nu)):
            raise Exception(f"Gn must have shape {(self.model.N, self.model.Nu)} but has shape {self.Gn.shape}")
        try:
            self.model.Enl(x)
        except:
            raise Exception(f"Enl function not working with input vector of size {self.model.N}")
        try:
            Fnl_val = self.model.Fnl(x)
        except:
            raise Exception(f"Fnl function not working with input vector of size {self.model.N}")
        if (Fnl_val.shape != J0x.shape):
            raise Exception(f"Fnl(q) must have {self.model.N} elements but has shape {Fnl_val.shape}")
        try:
            Enl_val, Fnl_val = self.model.EandFnl(x)
        except:
            raise Exception(f"EandFnl function not working with input vector of size {self.model.N}")
        if (Fnl_val.shape != J0x.shape):
            raise Exception(f"Fnl(q) evaluated using EandFnl(q) must have {self.model.N} elements but has shape {Fnl_val.shape}")
        
        try:
            self.model.setting()
        except:
            print(f"Model must have a setting() function")

    ### Energy related functions ### 

    def Ek_tilde(self, p):
        """Returns 0.5 * p^T Mtilde^{-1} p, correponding to the 
        kinetic part of the pseudo-energy preserved by the scheme.

        Args:
            p (vector): generalized momentum vector

        Returns:
            number: kinetic part of the pseudo-energy
        """
        return 0.5 * p.dot(1 / self.model.M * p 
                            - self.dt2 / 4 * self.model.J0 / self.model.M * self.model.K_op(self.model.J0 / self.model.M * p) 
                            - self.dt / (2 * self.model.M) * self.model.Rsv_op(p/self.model.M))
    
    def Ek(self, p):
        """Returns 0.5 * p^T M^{-1} p, correponding to the 
        kinetic part of the energy of the conitnuous time system.

        Args:
            p (vector): generalized momentum vector

        Returns:
            number: kinetic part of the continuous time energy
        """
        return 0.5 * p.dot(1 / self.model.M * p)
    
    def Ep_lin(self, q):
        """Returns 0.5 * q^T  K  q, correponding to the linear
        part of the potential energy.

        Args:
            q (vector): generalized coordinate vector

        Returns:
            number: linear part of the potential energy
        """
        return 0.5 * q.dot(self.model.K_op(q))

    def Ep_nl(self, q):
        """Returns the nonlinear part of the potential energy
        evaluated using the q corrdinates

        Args:
            q (vector): generalized coordinate vector

        Returns:
            number: nonlinear part of the potential energy evaluated from q
        """
        return self.model.Enl(q)

    def Ep_nl_sav(self, r):
        """Returns the nonlinear part of the potential energy
        evaluated using the scalar auxiliary variable

        Args:
            r (number): scalar auxiliary variable

        Returns:
            number: nonlinear part of the potential energy evaluated from r
        """
        return 0.5 * r * r

    def E_tilde(self, q, p, r):
        """Returns the preserved pseudo-energy of the scheme (eq 17)

        Args:
            q (vector): generalized coordinates mu_t+ q^{n-1/2}
            p (vector): momentum p^n
            r (_type_): scalar auxiliary variable r^n

        Returns:
            number: E^n
        """
        return self.Ek_tilde(p) + self.Ep_lin(q) + self.Ep_nl_sav(r)

    def E_orig(self, q, p):
        """Returns an evaluation of the original continuous time 
        energy evaluated from q and p

        Args:
            q (vector): generalized coordinates mu_t+ q^{n-1/2}
            p (vector): momentum p^n

        Returns:
            number: E^n
        """
        return self.Ep_lin(q) + self.Ep_nl(q) + self.Ek(p)
    
    def P_stored(self, Enow, Enext):
        """Return the energy variation P_stored^{n+1/2} given
        the energy E^{n+1} and E^n (eq 16)

        Args:
            Enow (number): pseudo-energy E^n
            Enext (number): pseudo-energy E^{n+1}

        Returns:
            number: P_stored^{n+1/2}
        """
        return (Enext - Enow) / self.dt

    def P_diss(self, pnow, pnext, Rmid):
        """Return the dissipated power P_diss^{n+1/2} given 
        p^n, p^{n+1} and R_{mid}^{n+1/2} (eq 16)

        Args:
            pnow (vector): momentum p^n
            pnext (vector): momentum p^{n+1}
            Rmid (vector): dissipation matrix R_{mid}^{n+1/2}

        Returns:
            number: P_diss^{n+1/2}
        """
        mutv = (pnow + pnext) / (2 * self.model.M)
        return mutv.dot( Rmid * mutv + self.model.Rsv_op(mutv))

    def P_ext(self, pnow, pnext, G, u):
        """Returns the power exchanged with the outer world through
        power ports P_ext^{n+1/2} given p^n, p^{n+1}, G^{n+1/2}, u^{n+1/2} (eq 16)
        The system receives energy for negative values.

        Args:
            pnow (vector): momentum p^n
            pnext (vector): momentum p^{n+1}
            G (matrix): input matrix G^{n+1/2}
            u (vector): input vector u^{n+1/2}

        Returns:
            number: P_ext^{n+1/2}
        """
        mutv = (pnow + pnext) / (2 * self.model.M)
        return -mutv.dot(G * u)

    def P_tot(self, Enow, Enext, pnow, pnext, Rmid, G, u):
        """Returns the discrete power balance. Should be zero up to
        numerical precision! (eq 16)

        Args:
            Enow (number): pseudo-energy E^n
            Enext (number): pseudo-energy E^{n+1}
            pnow (vector): momentum p^n
            pnext (vector): momentum p^{n+1}
            Rmid (_type_): dissipation matrix R_{mid}^{n+1/2}
            G (matrix): input matrix G^{n+1/2}
            u (vector): input vector u^{n+1/2}

        Returns:
            number: P_tot^{n+1/2}
        """
        return self.P_stored(Enow, Enext) + self.P_diss(pnow, pnext, Rmid) + self.P_ext(pnow, pnext, G, u) 

    ### Intermediary solver functions ###

    def A0_inv(self, Rmid):
        """Recomputes A0_inv used in Shermann-Morrison solving 
        (eq 19h) from the expression of Rmid.

        Args:
            Rmid (vector): diagonal dissipation matrix Rmid

        Returns:
            vector: diagonal matrix A0_inv
        """
        return 2 * self.dt2 * np.ones(self.model.N) / (2 * self.model.M + self.dt * Rmid)
    
    def B_op(self, q):
        """Applies the B operator to the vector q^{n+1/2} (eq 19b, 19e)

        Args:
            q (vector): generalized coordinates vector

        Returns:
            vector: B q
        """
        return 2 * self.model.M / (self.dt2 * self.model.J0) * q - self.model.J0 * self.model.K_op(q) - self.model.Rsv_op(q / self.model.J0) / self.dt
    
    def C_op(self, q, g, Rmid):
        """Applies the C operator to the vector q^{n-1/2} (eq 19b, 19f)

        Args:
            q (vector): generalized coordinates vector
            g (vector): g vector (eq 19a)
            Rmid (vector): diagonal dissipation matrix Rmid

        Returns:
            vector: C q
        """
        return - np.ones(self.model.N) * \
            (
                (self.model.M / self.dt2
                - Rmid / (2 * self.dt)) 
                * q / self.model.J0
                - 0.25 * g * g.dot(q / self.model.J0)
                - self.model.Rsv_op(q / self.model.J0) / self.dt
            )
                                              
    def g(self, q):
        """Return the g vector computed from q as (eq 9b)
        J0 fnl(q) / sqrt(2 * Enl(q) + C0)

        Args:
            q (vector): generalized coordinates vector

        Returns:
            vector: g_std(q)
        """
        Enl, Fnl = self.model.EandFnl(q)
        return self.model.J0 * Fnl / (np.sqrt(2 * Enl + self.C0 + self.Num_eps))
    
    def g_mod(self, q, p, r):
        """Returns the modification of g term to include 
        the drift controller

        Args:
            q (vector): generalized coordinates vector
            p (vector): momentum vector
            r (number): scalar auxiliary variable

        Returns:
            vector: g_mod(q,p, r)
        """
        self.epsilon = r - np.sqrt(2 * self.model.Enl(q) + self.C0)
        return - self.lambda0 * self.epsilon * self.model.M * np.sign(p) / (np.abs(p) + self.Num_eps)

    ### Time stepping and integration ###

    def time_step(self, qlast, qnow, rn, unow, ConstantRmid = False):
        """Returns next state by solving (eq 19a, 19b, 19c)

        Args:
            qlast (vector): generalized coordinates (q^{n-1/2})
            qnow (vector): generalized coordinates (q^{n+1/2})
            rn (_type_): scalar auxiliary variable (r^n)
            unow (vector): input
            ConstantRmid (bool, optional): Specifies if Rmid needs to be recomputed,
                which corresponds to cases where it is state dependent. Defaults to False.

        Returns:
            (vector, number): next state (q^{n+3/2}, r^{n+1})
        """
        # qlast correponds to q^{n-1/2} and qnow to
        # q^{n+1/2}. rn corresponds to r^n. unow to u^{n+1/2}.
        # First, compute g
        pn = (qnow - qlast) * self.M_J0dt
        qn = (qlast + qnow)/2
        self.gn = self.g(qnow) + self.g_mod(qn, pn, rn)

        # Compute Rmid and G
        self.Gn = self.model.G(qnow)
        if not ConstantRmid:
            self.Rmidn = self.model.Rmid(qnow)
            # Get qnext q^{n+3/2} using Shermann-Morrison
            self.A0_inv_n = self.A0_inv(self.Rmidn)

        den = (4 + self.gn.dot(self.A0_inv_n * self.gn)) # eq 19g

        self.RHSn = self.B_op(qnow) + self.C_op(qlast, self.gn, self.Rmidn) - self.gn * rn + self.Gn @ unow # eq 19b
        
        qnext = self.model.J0 * self.A0_inv_n * self.RHSn \
            - self.model.J0 * self.A0_inv_n * self.gn / den * self.gn.dot(self.A0_inv_n * self.RHSn) # 19b+19g
        
        # Update auxiliary variable
        rnext = rn + self.gn.dot((qnext - qlast) / (2 * self.model.J0))
        return qnext, rnext, qn, pn, self.epsilon
    
    def integrate(self, q0, u0, u_func, duration, ConstantRmid = False,
                  storage_config = DEFAULT_STORAGE_CONFIG, plotter_config = DEFAULT_PLOTTER_CONFIG):
        """Integrates the dynamics using provided initial conditions, input function, 
        and duration.

        Args:
            q0 (vector): Initial condition, generalized coordinates
            u0 (vector): Initial condition, velocity u_0 = dot q(t=0)
            u_func (function): input vector as a function of time
            duration (number): simulation duration
            plot (int, optional): Specifies which data to plot. Defaults to None.
            ConstantRmid (bool, optional): Specifies if Rmid needs to be recomputed at each time step. Defaults to False.
        """
        ### Time vector and storage initialization ###
        self.model.Nt = int(duration / self.dt)
        self.t = np.arange(self.model.Nt) * self.dt

        self.storage = ResultsStorage(**storage_config)
        self.storage.reserve(self.t, self.model, self)
        self.plotter = Plotter(**plotter_config)
        self.plotter.init_plots()

        ### State initialization ###
        qlast = q0 - u0 * self.dt / 2
        qnow = q0 + u0 * self.dt / 2
        rnow = np.sqrt(2 * self.model.Enl(q0) + self.C0)

        if ConstantRmid:
            # Rmid vector is evaluated once at the begining of the simulation in this case
            self.Rmidn = self.model.Rmid(q0)
            self.A0_inv_n = self.A0_inv(self.Rmidn)

        ### Main loop ###
        for i in range(self.model.Nt ):
            qnext, rnext, qn, pn, epsilon = self.time_step(qlast, qnow, rnow, u_func((i+0.5) * self.dt), ConstantRmid=ConstantRmid)
            
            self.storage.store(q = qn, p = pn, r= rnow,
                           epsilon = epsilon, i = i, Rmid=self.Rmidn,
                            G = self.Gn, u = u_func((i+0.5) * self.dt), solver=self)
            self.plotter.update_plots(self.storage, block=False)

            qlast = qnow
            qnow = qnext
            rnow = rnext
        self.plotter.update_plots(self.storage, block=True)

if __name__ == "__main__":
    model = Model(10)
    solver = SAVSolver(model, sr = 100)
    solver.check_sizes()
    q0 = np.ones(model.N)
    u0 = np.zeros(model.N)
    def u_func(t):
        return np.zeros(model.Nu)
    solver.integrate(q0, u0, u_func, 100)


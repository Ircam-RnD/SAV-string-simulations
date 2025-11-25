import numpy as np

class Model():
    """General Model class to pass to the SAVSolver.
    """
    def __init__(self, N = 10):
        # Number of dofs
        self.N = N
        # Number of inputs
        self.Nu = 1
        # Diagonal system matrices
        self.J0 = np.ones(self.N)
        self.M = self.J0
        # Rmid is diagonal but can depend on the state
        self.Rmid = lambda q: (1 * np.ones(self.N))
        # Non-diagonal system matrices as operators
        self.K_op = lambda x: (x)
        self.Rsv_op = lambda x: (0 * x)
        # Input matrix, may depend on the state
        self.G = lambda q: (np.ones((self.N, self.Nu)))
        # Nonlinear functions
        self.Enl = lambda q: (0.25 * np.sum(q**4)) * 1
        self.Fnl = lambda q: (q**3) * 1
        # Both functions are called at the same time in the solver,
        # in some cases it is then computationally interesting 
        # to compute both in the same function.
        self.EandFnl = lambda q: (self.Enl(q), self.Fnl(q))
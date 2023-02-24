import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from scipy.fftpack import fft,ifft
from matplotlib import rc
rc('animation', html='jshtml')

#from google.colab import drive
#drive.mount('/content/drive')

class Schrodinger:
    def __init__(self, x, psi_x0, V_x, k0 = None, hbar = 1, m=1, t0=0.0):
        #x = xaxis array of length N giving position
        #V_x = yaxis array of length N giving potential
        #psi_x0 = array of length N giving intial wave function at t0
        #k0 gives the minimumum value of the momentum (there are some constraints on this due to the FFT: k0<k<2pi/dx where dx = x[1]-x[0])
        #default hbar = 1 and mass = 1 and initial time = 0
        self.x, psi_x0, self.V_x = map(np.asarray, (x, psi_x0, V_x))
        N = self.x.size
        assert self.x.shape == (N,)
        assert psi_x0.shape == (N,)
        assert self.V_x.shape == (N,)
        self.hbar = hbar
        self.m = m
        self.t = t0
        self.dt_ = None
        self.N = len(x)
        self.dx = self.x[1] - self.x[0] 
        #setting the pos steps (similar to doing dx = b-a/N as coordinates are evenly spaced)
        self.dk = 2 * np.pi / (self.N * self.dx) 
        #dk = 2pi/Ndx (do this so FFT looks like continuous fourier transform)
        #dk = 2pi/b-a
        
        if k0 == None:
            self.k0 = -0.5 * self.N * self.dk
        else:
            self.k0 = k0
        self.k = self.k0 + self.dk * np.arange(self.N) 
        # k = k0 + dk
        self.psi_x = psi_x0
        
        self.psi_x = psi_x0
        
        #variables which hold steps in evolution
        self.x_evolve_half = None
        self.x_evolve = None
        self.k_evolve = None
        
        #attributes used for dynamic plotting
        self.psi_x_line = None
        self.psi_k_line = None
        self.V_x_line = None
        
    def _set_psi_x(self, psi_x):
        #brings it back to the original continuous version of psi(x)
        self.psi_discrete_x = (psi_x * np.exp(-1j * self.k[0] * self.x) * self.dx / np.sqrt(2 * np.pi))
        
    def _get_psi_x(self):
        #gives the discrete vesion of psi needed for the FFT
        return(self.psi_discrete_x * np.exp(1j * self.k[0] * self.x) * np.sqrt(2 * np.pi) / self.dx)
        
#     def _set_psi_k(self, psi_k):
#         self.psi_mod_k = psi_k * np.exp(1j * self.x[0] * self.dk * np.arange(self.N))

#     def _get_psi_k(self):
#         return self.psi_mod_k * np.exp(-1j * self.x[0] * self.dk * np.arange(self.N))
# #can uncomment this stuff if we want to animate in momentum space too

    def _get_dt(self):
        return self.dt_

    def _set_dt(self, dt):
        #how we evolve psi - half steps in position space and full steps in momentum space
        #as we will see in the strang splitting method below
        if dt != self.dt_:
            self.dt_ = dt
            self.x_evolve_half = np.exp(-0.5 * 1j * self.V_x / self.hbar * dt ) 
            self.k_evolve = np.exp(-0.5 * 1j * self.hbar / self.m * (self.k * self.k) * dt)  
    
    psi_x = property(_get_psi_x, _set_psi_x)
#     psi_k = property(_get_psi_k, _set_psi_k)
    dt = property(_get_dt, _set_dt)
    #dont really understand property functions well
    #from what I understand, property assigns getter and setter functions to a variable
    
    def time_step(self, dt, Nsteps = 1):
        self.dt = dt
        
        #strang splitting:
        for i in range(Nsteps):
            #half step in position:
            self.psi_discrete_x *= self.x_evolve_half
            #FFT
            self.psi_discrete_k = fft(self.psi_discrete_x)
            #full step in momentum
            self.psi_discrete_k *= self.k_evolve
            #iFFT
            self.psi_discrete_x = ifft(self.psi_discrete_k)
            #half step in position
            self.psi_discrete_x *= self.x_evolve_half
        
        self.t += dt * Nsteps
        #t = t + Nsteps*dt
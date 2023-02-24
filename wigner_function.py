import numpy as np
from matplotlib import pyplot as plt, cm, colors
from scipy.fftpack import fft,ifft
from scipy.ndimage import shift
from matplotlib.ticker import LinearLocator
from mpl_toolkits.mplot3d import Axes3D

def wigner_plot(psi_x0, x, y, dx, dy, k0_ft_y):
    x, y = map(np.asarray, (x, y))
    
    #setting the values of psi(x+y)
    psi_x0_plus = np.zeros((len(y),len(x)), dtype=complex)
    for i in range(len(y)):
        psi_x0_plus[i][:] = shift(psi_x0, int(-1/dy)*y[i], cval=0.0)
    
    #setting the values of psi*(x-y)
    psi_x0_minus = np.zeros((len(y),len(x)), dtype=complex)
    for i in range(len(y)):
        psi_x0_minus[i][:] = np.conjugate(shift(psi_x0, int(1/dy)*y[i], cval=0.0))
        
    #multiplying the two wavefunction together so our function is of the form psi(x+y)psi*(x-y)
    wigner_to_transform = np.multiply(psi_x0_plus,psi_x0_minus)

    #discretising the function so we can fourier transform correctly
    discrete_wigner_to_tranform = np.zeros((len(y),len(x)), dtype = complex)
    for i in range(len(y)):
        discrete_wigner_to_tranform[i][:] = wigner_to_transform[i] * np.exp(-1j * k0_ft_y * y[i]) * dy / np.sqrt(2 * np.pi)

    #fast fourier transforming
    discrete_fourier_wigner = fft(discrete_wigner_to_tranform, axis = 0)

    #un-discretising the function
    wigner = np.zeros((len(y),len(x)), dtype = complex)
    for i in range(len(y)):
        wigner[i][:] = (discrete_fourier_wigner[i] * np.exp(1j * k0_ft_y * y[i]) * np.sqrt(2 * np.pi) / dy)
     
    #checking all the values of the wigner function are real:
    assert np.allclose(wigner.imag, 0, rtol=0, atol=1e-9), 'wigner has non-zero imaginary components' 
    
    wigner = np.real(wigner)
    
    #plotting
    #3D surface plot
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    y_smaller = y[int(79*(1/dy)):int(-80*(1/dy))]
    x_smaller = x[int(24*(1/dx)):int(-25*(1/dx))]
    X, Y = np.meshgrid(x_smaller,y_smaller)
    Z = wigner[int(79*(1/dy)):int(-80*(1/dy)),int(24*(1/dx)):int(-25*(1/dx))]
    ax.set_xlabel('x')
    ax.set_ylabel('k')
    ax.set_zlabel('wigner')
    surf = ax.plot_surface(X,Y,Z)
    fig.colorbar(surf, shrink = 0.25)
    ax.view_init(azim = 230)
    
    #density plot
    fig = plt.figure(figsize = (6,5))
    plt.pcolormesh(X, Y, Z, shading='auto')
    plt.clim(-np.max(wigner), np.max(wigner))
#     plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('k', rotation = 0)
    plt.show()
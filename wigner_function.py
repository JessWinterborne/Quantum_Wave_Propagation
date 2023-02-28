import numpy as np
from matplotlib import pyplot as plt, cm, colors
from scipy.fftpack import fft,ifft
from scipy.ndimage import shift
import os

def wigner_plot(psi_x0, x, y, dx, dy, k0_ft_y, t_max, view, limit, save_fig = False, frame_num = None):
    if save_fig == True:
        assert frame_num is not None, "Please provide framenumber if saving"
        
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
    
    #creating a folder for the frames
    #we will use these frames to animate
    if save_fig == True:
        try:
            os.mkdir(os.path.join(os.getcwd(),'wigner_frames'))
        except FileExistsError:
            print("Frames folder found, overwritting")
    
    #plotting:
    
    #3D surface plot
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    y_smaller = y[int(79*(1/dy)):int(-80*(1/dy))]
    x_smaller = x[int(24*(1/dx)):int(-25*(1/dx))]
    X, Y = np.meshgrid(x_smaller,y_smaller)
    Z = wigner[int(79*(1/dy)):int(-80*(1/dy)),int(24*(1/dx)):int(-25*(1/dx))]
    ax.set_zlim3d(-limit, limit)
    ax.set_zticks(np.arange(-limit, limit+1, 5.0))
    ax.set_xlabel('q')
    ax.set_ylabel('p')
    ax.set_zlabel('W(q,p)')
    if frame_num is not None:
        if frame_num == 0:
            ax.set_title('t=%.2f' %(0))
        else:
            ax.set_title('t=%.2f' %(10*(int(frame_num)/int(t_max))))
    surf = ax.plot_surface(X,Y,Z)
#     fig.colorbar(surf, shrink = 0.25)
    ax.view_init(azim = view)
    
    #saving the frame
    if save_fig == True:
        plt.savefig(f'wigner_frames/3d_{frame_num}', dpi=300)
    
    #density plot
    fig = plt.figure(figsize = (6,5))
    plt.pcolormesh(X, Y, Z, shading='auto')
    plt.clim(-limit, limit)
    cbar = plt.colorbar()
    cbar.set_label('W(q,p)')
    plt.xlabel('q')
    plt.ylabel('p', rotation = 0)
    if frame_num is not None:
        if frame_num == 0:
            plt.title('t=%.2f' %(0))
        else:
            plt.title('t=%.2f' %(10*(int(frame_num)/int(t_max))))
   
    #saving the frame
    if save_fig == True:
        plt.savefig(f'wigner_frames/density_{frame_num}', dpi=300)
        
    plt.show()
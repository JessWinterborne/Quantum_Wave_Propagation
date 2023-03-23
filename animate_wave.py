import numpy as np
import os
from matplotlib import pyplot as plt
from matplotlib import animation
from scipy.fftpack import fft,ifft
from matplotlib import rc
rc('animation', html='jshtml')

def save_wave(xlim, ylim, S, frames, dt, N_steps, name, scaling, save = True):
    fig = plt.figure(figsize = (10,8), dpi=300)
    ax = fig.add_subplot(111, xlim=xlim, ylim=ylim)
    #have subplot so we can add in a plot for the momentum if we wanted (see below)
    psi_x_line, = ax.plot([], [], c='r', label=r'$|\psi(q)|^2$')
    # psi_k_line, = ax2.plot([], [], c='r', label=r'$|\psi(k)|$')
    V_x_line, = ax.plot([], [], c='k', label=r'$V(q)$')
    title = ax.set_title('')
    ax.legend(prop = dict(size=12))
    ax.set_xlabel('$q$')
    ax.set_ylabel(r'$|\psi(q)|^2$')
    V_x_line.set_data(S.x, S.V_x)
    def init():
        psi_x_line.set_data([], [])
        V_x_line.set_data([], [])
        return (psi_x_line, V_x_line)
    if save == True:
        try:
            os.mkdir(os.path.join(os.getcwd(),'wave_frames'))
        except FileExistsError:
            print("Frames folder found, overwritting")
    def animate(i):
        psi_x_line.set_data(S.x, scaling*abs(S.psi_x)**2)
        #factor in front just scales the wave function up
        V_x_line.set_data(S.x, S.V_x)
        title.set_text('t=%.2f' %S.t)
        if save == True:
            plt.savefig('wave_frames/wave_frame=%2f.png'%i)
        S.time_step(dt, N_steps)
        return (psi_x_line, V_x_line)
    
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=frames, interval=40, blit=True)
    if save == True:
        anim.save(name+'.mp4')
    else:
        return(anim)
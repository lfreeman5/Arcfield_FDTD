from fdtd_2d_function import fdtd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft

def sinusoidal(t,_):
    return 50*np.sin(1e9*t)

def deriv_gaussian(t,dt):
    spread = 220*dt
    t0 = 600*dt
    amplitude = 100
    return -amplitude*(t0-t)/spread*np.exp(-((t0-t)/spread)**2)

if __name__ == "__main__":
    nx,ny = 500,500
    dx = 2.5e-4
    pxstart = 200
    pystart = 200
    wp = 2*np.pi*8e1*np.ones((1,1))
    nu = 1e1*np.ones((1,1))
    nsteps = 5000
    input_slices = [(slice(10,11),slice(0,None))]
    poll_slices = [(slice(0,None),slice(int(ny/2),int(ny/2)+1))]

    polled_data, dt = fdtd(nx,ny,dx,pxstart,pystart,wp,nu,deriv_gaussian,nsteps,input_slices,poll_slices,last_input_timestep=1800)
    middle_slice = polled_data[0]


    """Data comparison w/1D and exact"""
    print(f'Shape of middle slice: {middle_slice.shape}')
    input_data = middle_slice[:,10,0].copy()
    reflected_data = middle_slice[:,10,0].copy()
    transmitted_data = middle_slice[:,1210,0].copy()

    transition_timestep = 1800
    input_data[transition_timestep:] = 0.0
    reflected_data[:transition_timestep] = 0.0
    transmitted_data[:transition_timestep] = 0.0

    sampling_frequency = 1/dt
    frequencies = np.linspace(0,sampling_frequency,nsteps)

    """Input,Refl,Trans results"""
    # Create a figure and subplots
    fig, axs = plt.subplots(3, 1, figsize=(8, 10))  # 3 rows, 1 column

    # Plot data on each subplot and set titles and x-limits
    axs[0].plot(input_data, color='blue')
    axs[0].set_title('Input Data')
    # axs[0].set_xlim([0, 10])  # Example x-limits

    axs[1].plot(reflected_data, color='green')
    axs[1].set_title('Reflected Data')
    # axs[1].set_xlim([0, 10])  # Example x-limits

    axs[2].plot(transmitted_data, color='red')
    axs[2].set_title('Transmitted Data')
    # axs[2].set_xlim([0, 10])  # Example x-limits
    plt.tight_layout()
    plt.show()


    """Input signal PSD"""
    plt.plot(frequencies/1.e9, np.abs(fft(input_data)*np.conj(fft(input_data))))
    plt.xlim(0,10)
    plt.title('Power Spectral Density of Input Signal')
    plt.show()

    """Reflected Coefficient"""
    plt.scatter(frequencies/1.e9, 10.*np.log10(np.abs(fft(reflected_data)/fft(input_data))))
    plt.xlim(0,10)
    plt.title('Reflected Coefficient')
    plt.show()

    """Transmission Coefficient"""
    plt.scatter(frequencies/1.e9, 10.*np.log10(np.abs(fft(transmitted_data)/fft(input_data))))
    plt.title('Transmission Coefficient')
    plt.xlim(0,10)
    plt.show()
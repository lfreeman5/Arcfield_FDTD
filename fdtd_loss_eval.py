import numpy as np
from numpy.fft import fft
import cmath
import matplotlib.pyplot as plt
import time
from joblib import Parallel, delayed


"""Physical Constants"""
c_0 = 2.9975e+8 # speed of light in m/s
eps_0 = 8.854e-12 # permittivity of free space in F/m
mu_0 = 1.257e-6 # permeability of free space in H/m

"""Program Constants"""
N_STEPS = 25000
CFL = 0.5
dx = 0.00015

def fft_extract(inp, trans, dt, freq):
    f_inp = fft(inp)
    f_trans = fft(trans)
    # f_refl = fftn(refl)
    f_s = 1/dt # Sampling Frequency in Hz
    f_range = np.linspace(0,f_s,len(f_inp))
    trans_coeffs = 10 * np.log10(np.abs((f_trans/f_inp)**2))
    # refl_coeffs = 10 * np.log10(np.abs((f_refl/f_inp)**2))
    return np.interp(freq, f_range, trans_coeffs)

def calc_loss(plasma, coll, frequency,x_spacing):
    """
    Calculates the transmision loss in dB for a signal of a given frequency passing through a plasma field 
    plasma is a numpy Nx1 array of plasma frequencies along a ray (Hz)
    coll is a numpy Nx1 array of collision frequencies along a ray (Hz)
    spacing is the cell size, defined for both plasma and coll (m)
    frequency is the radio frequency to evaluate the loss at (Hz)
    x_spacing is a linspace of the spacing of the grid in meters
    """
    ss = int(x_spacing[-1]/dx)
    # print(f'SS: {ss}') 
    x_fdtd = np.linspace(0,x_spacing[-1],ss)
    # print(plasma)
    plasma = np.interp(x_fdtd,x_spacing,plasma)
    # print(plasma)
    coll = np.interp(x_fdtd,x_spacing,coll)
    # dx = x_fdtd[1] # Should be very close to dx but they're not exactly the same.
    assert(len(plasma)==len(coll))
    n_pts = ss*4 + 1 # The number of points is equal to the number of cells (9*plasma) plus 1
    dt = CFL*dx/c_0
    ez, ez_0 = np.zeros(n_pts,dtype = complex), np.zeros(n_pts,dtype = complex)
    hy = np.zeros(n_pts-1,dtype = complex)
    inp,trans,refl = np.zeros(N_STEPS), np.zeros(N_STEPS), np.zeros(N_STEPS)
    lbc_1, lbc_2 = 0.,0.
    rbc_1,rbc_2 = 0.,0.    

    ### Set up derivative gaussian
    spread = 220*dt
    t0 = 600*dt
    amplitude = 100
    def pulse(t):
        return -amplitude * (t0-t) / spread * np.exp(-((t0-t)/spread)**2)

    ### Calculate Matrix-Exponential FDTD parameters
    wp, nu = np.pi*2*np.maximum(plasma,1), np.maximum(coll,1)
    a = np.sqrt(nu**2-4*wp**2,dtype=complex)/2.
    b = nu/2.
    D = (a+b)/(2*a) * np.exp((a-b)*dt) + (a-b)/(2*a) * np.exp((-a-b)*dt)
    F = (np.exp((a-b)*dt) - np.exp((-a-b)*dt))/(2*a)
    B = (a+b) / (2*a*(a-b)) * (np.exp((a-b) * dt)-1) - (a-b)/(2*a*(a+b))*(np.exp((-a-b)*dt)-1)
    C = 1/(2*a*(a-b))*(np.exp((a-b)*dt)-1) + 1/(2*a*(a+b))*(np.exp((-a-b)*dt)-1)
    j = np.zeros_like(F,dtype = complex)

    ### Main loop
    for n in range(N_STEPS):
        # Store old E_Z field
        ez_0 = ez.copy()
        # Update ez
        ez[1:-1] = ez_0[1:-1] + dt/(eps_0*dx)*(hy[1:]-hy[:-1])
        ez[2*ss:3*ss] = D * ez_0[2*ss:3*ss] + B/(eps_0*dx) * (hy[2*ss:3*ss] - hy[2*ss-1:3*ss-1]) - F/eps_0*j
        j = j*(D-nu*F+wp**2*C*F/B) + eps_0*wp**2*C/B*ez[2*ss:3*ss] + (eps_0*wp**2*F - eps_0*wp**2*C*D/B)*ez_0[2*ss:3*ss]
        # Add gaussian pulse
        t = n*dt
        ez[10] += pulse(t)

        # ABCs on Left/Right side
        ez[0] = lbc_2
        lbc_2,lbc_1 = lbc_1,ez[1]
        ez[-1] = rbc_2
        rbc_2,rbc_1 = rbc_1,ez[-2]
        # Update hy
        hy = hy + dt/(mu_0*dx) * (ez[1:]-ez[:-1])
        if(t<2*ss*dx/c_0):
            inp[n]=ez[10].real
        else:
            refl[n]=ez[10].real
            trans[n] = ez[3*ss+10].real
    return fft_extract(inp, trans, dt, frequency)


if __name__ == "__main__":
    # import cProfile
    start = time.time()
    N_EXEC = 50
    test_plasma = 4e9*np.ones(400)
    test_coll = 1e9*np.ones(400)
    # cProfile.run('calc_loss(test_plasma,test_coll,1e9)')

    for i in range(N_EXEC):
        calc_loss(test_plasma, test_coll, 1e9, np.linspace(0,0.1,400))
    end = time.time()
    print(f'Elapsed time: {end-start}s, average: {(end-start)/N_EXEC}')
    

    # import os
    # from PIL import Image, ImageSequence

    # directory = './plots/'

    # files = os.listdir(directory)

    # image_files = sorted([os.path.join(directory, file) for file in files if file.startswith('plot_') and file.endswith('.png')],
    #                     key=lambda x: int(x.split('_')[1].split('.')[0]))

    # images = [Image.open(file) for file in image_files]

    # gif_file = './plots/plots.gif'

    # images[0].save(gif_file, save_all=True, append_images=images[1:], duration=200, loop=0)

    # print(f'GIF saved to {gif_file}')
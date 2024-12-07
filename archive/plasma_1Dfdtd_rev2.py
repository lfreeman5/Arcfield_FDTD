# ******
# One-dimensional Matrix Exponential Finite Difference Time Domain 
# (ME-FDTD) electromagnetic propagation code
#
# 24 May 2024. Meredith Amrhein
#
# Majority of code moved from MatLab file plasma_1d_me_fdtd_rev10.m made
# by Russ Adelgren
#
# The purpose of this 1D ME-FDTD EM code is to calculate the attenuation
# through a spatially varying plasma sheath.  The code is set up to read 
# in US3D output with the plasma frequency and collision frequency given
# along a propagation axis.  The code calculates the reflection and
# transmission coefficients for the electric field caused by the plasma
# field.  The transmission coefficient in turn can be used to calculate the
# signal loss from attenuation in the plasma field.
#
# References:
# 1.  Liu, S., Liu, Sh., "Simulation of Electromagnetic Wave Propagation 
#     in Plasma Using Matrix Exponential FDTD Method," Journal of Milli
#     Terahertz Waves, Springer Science, 2009.
# 2.  Zivanovic, S.V., Musal, H.M., "Determination of Plasma Layer
#     Properties from the Measured Electromagnetic Transmission
#     Coefficient," IEEE Transactions on Antennas and Propagation,
#     September 1964.
# 3.  Inan, U.S., Marshall, R.A., "Numerical Electromagnetics - The FDTD 
#     Method (book)," Cambridge University Press, 2011.
# 4.  Adamy, D., "EW 101 - A First Course in Electronic Warfare (book),"
#     Artech House, 2001.
#
# All units are SI metric (m-kg-s) unless explicitly stated otherwise.
#
# ******
#
##
import math
import cmath
from Plasma_FDTD_Functions_rev1 import *
from scipy.interpolate import interp1d
from scipy.fft import fft, ifft

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation 
import textwrap
#import pandas as pd
#def plasma_1d_fdtd_rev1(data, fxmt, stamp, filename_out, tfreq):
Dimention = 1
Want2Animate = 0
NOdata = False
#Any graphs you do not want set the value at that index to 0
#Index 0: Plasma and Collision Frequency graph
#Index 1: Time Domain Signals (Input, Reflected and Transmitted)
#Index 2: Derivative Gaussian Pulse Distribution
#Index 3: Reflected and Transmission Coefficients
Figures2Print = [0,0,0,0]
# 0: normal run with plasma
#1: normal run without plasma
# 2: normal run with plasma all variable 2D to see if 2D lists mess with calculations
TestLoop = 2
filename = "Data1D"
ddx = 0.00025 # 0.008
dd = [ddx,0,0]
L=0.4
Kx=L/ddx
KE=[Kx,0,0]
nsteps=10000.0 
eps_0=8.854*(10**-12)
mu_0=4*math.pi*(10**-7) 
c_0 = 2.9975*(10**8) 
CFL = 0.5 
dt = CFL*ddx/c_0
kstart=(0.2/ddx) 
kend=(.3/ddx)
plasmastart = [kstart,0,0]
plasmaend = [kend,0,0]
increment = 0.1

ez=[0]*round(Kx+1)
ez_0=[0]*round(Kx+1)
j=[0]*round(Kx+1) 
hy=[0]*round(Kx+1) 
geninput=[0]*int(nsteps) 
trans=[0]*int(nsteps) 
refl=[0]*int(nsteps) 
t = []
for i in range(0,int(nsteps)):
    t.append(i) #i*dt
i=0
probe_input = 10.0 
probe_reflected = 0.1/ddx-10 
probe_transmitted = .3/ddx+10 
wpHz_plasma = 4*(10**9)
nuHz_plasma = 1*(10**9)

wHz_exact = []
w = []
lam = []
kList = []
Op = []
Oc = []
Np = []
gamp = []
T_exact = []
magT_exact = []
R_exact = []
magR_exact = []

#Initial conditions for crude absorbing boundaries
ez_low_m1=0.0
ez_low_m2=0.0
ez_high_m1=0.0
ez_high_m2=0.0
T=0


# Read in Tecplot post processed US3D data
timet='4.0'
listOfColumns = []
data = []
if NOdata == False:
    listOfColumns,data = dataHandling(filename)
else:
    #faking data values while debugging because these will be replaced later
    #with the wpHz_plasma and nu_plasma
    listOfColumns = colNumList = [0,1,2,3,4]
    data = [[1]*5]*30
wpHz, nu, xint, wpHzy, nuy, yint, wpHzz, nuz, zint = WpNuSetup(Dimention,NOdata,data, listOfColumns, wpHz_plasma, nuHz_plasma, plasmastart,plasmaend,KE,dd)
x = np.linspace(0,L,int(Kx),endpoint=False)
# Plot the plasma input data
plt.semilogy(xint[:-1],wpHz, color = 'r')
plt.semilogy(xint[:-1],nu, color = 'b')
plt.xlabel("Propagation distance (m)")
plt.ylabel("Frequency (Hz)")
plt.legend(['Plasma','Collision'])
plt.xlim(0.001 , L)
plt.ylim(1 , 1*(10**11))
plt.grid()
plt.semilogy(kstart*ddx,1, color = 'r', marker = 'o')
plt.semilogy(kend*ddx,1, color = 'b', marker = 'o')
plt.show()
# Convert plasma frequency to angular frequency
wp = [1]*len(wpHz)
for i in range(0,len(wpHz)):
    wp[i] = 2*math.pi*wpHz[i]
##
# Set up conditions for derivative Gaussian input signal 
spread = 220*dt
t0 = 600*dt
amplitude = 100
wHz_exact , magR_exact, magT_exact = exact(kstart,kend,ddx,c_0,wpHz_plasma,nuHz_plasma)
##
# Calculate ME-FDTD parameters
# a= []
# b= []
# D= []
# F= []
# B= []
# C= []
# for i in range(0,len(wp)):
#     a.append((cmath.sqrt((nu[i]**2)-(4*(wp[i]**2))))/2) # This piece of eigenvalue can be complex ...
#     b.append(nu[i]/2)
#     D.append((a[-1]+b[-1])/(2*a[-1])*cmath.exp((a[-1]-b[-1])*dt) + ((a[-1]-b[-1])/(2*a[-1])*cmath.exp((-a[-1]-b[-1])*dt)))
#     F.append((cmath.exp((a[-1]-b[-1])*dt) - cmath.exp((-a[-1]-b[-1])*dt))/(2*a[-1]))
#     B.append((a[-1]+b[-1])/(2*a[-1]*(a[-1]-b[-1]))*(cmath.exp((a[-1]-b[-1])*dt)-1) - (a[-1]-b[-1])/(2*a[-1]*(a[-1]+b[-1]))*(cmath.exp((-a[-1]-b[-1])*dt)-1))
#     C.append(1/(2*a[-1]*(a[-1]-b[-1]))*(cmath.exp((a[-1]-b[-1])*dt)-1) + 1/(2*a[-1]*(a[-1]+b[-1]))*(cmath.exp((-a[-1]-b[-1])*dt)-1))

### Calculate Matrix-Exponential FDTD parameters
# wp, nu = np.pi*2*np.maximum(plasma,1), np.maximum(coll,1)
a = np.sqrt(nu**2-4*wp**2,dtype=complex)/2.
b = nu/2
D = (a+b)/(2*a) * np.exp((a-b)*dt) + (a-b)/(2*a) * np.exp((-a-b)*dt)
F = (np.exp((a-b)*dt) - np.exp((-a-b)*dt))/(2*a)
B = (a+b) / (2*a*(a-b)) * (np.exp((a-b) * dt)-1) - (a-b)/(2*a*(a+b))*(np.exp((-a-b)*dt)-1)
C = 1/(2*a*(a-b))*(np.exp((a-b)*dt)-1) + 1/(2*a*(a+b))*(np.exp((-a-b)*dt)-1)
# j = np.zeros_like(F,dtype = complex)
#VIDEO VARIABLES
#animatedPlot = [[1]*round(Kx+1)]*int(nsteps) 
animatedPlot = []


######Main Loop Begins######
#tic; 
# Outer time step loop
if TestLoop == 0:
    for n in range(0,int(nsteps)): 
        if n%100 == 0:
            print("this is n %0.0f" %n)
        # Spatial loop up to front edge of plasma for Ez field
        # k=2
        # while k<=int(kstart-1):
        #     ez[k] = ez[k] + dt/eps_0/ddx*(hy[k]-hy[k-1]);
        #     k+= 1
        ez[2:int(kstart)] = ez[2:int(kstart)] + dt/eps_0/ddx*(hy[2:int(kstart)]-hy[1:int(kstart-1)]);
        
        # Source Pulse
        pulse = -amplitude*(t0-T)/spread*np.exp(-((t0-T)/spread)**2) # Derivative Gaussian input signal
        #pulse = 40*sin(2*pi*3e9*T) # Sine input with single frequency
        ez[10]=ez[10]+pulse
        T+=dt
        
        # Spatial loop through plasma with ME-FDTD formulation for Ez field
        # k=int(kstart)
        # end = int(kend)
        # while k<=end:
        #     ez[k] = D[k]*ez[k] + B[k]/eps_0/ddx*(hy[k]-hy[k-1]) - F[k]/eps_0*j[k]
        #     j[k] = (D[k] - nu[k]*F[k] + (wp[k]**2)*C[k]*F[k]/B[k])*j[k] + (eps_0*(wp[k]**2)*C[k]/B[k])*ez[k] +\
        #         (eps_0*(wp[k]**2)*F[k] - eps_0*(wp[k]**2)*C[k]*D[k]/B[k])*ez_0[k]
        #     ez_0[k] = ez[k] # Store current time step Ez field for next iteration
        #     k+= 1
        #     if n >=400 and n<1000:
        #         print(f'This is j: {j[k]}')
        ez[int(kstart):int(kend+1)] = D[int(kstart):int(kend+1)]*ez[int(kstart):int(kend+1)] + \
            B[int(kstart):int(kend+1)]/eps_0/ddx*(hy[int(kstart):int(kend+1)]-hy[int(kstart-1):int(kend)]) - \
                F[int(kstart):int(kend+1)]/eps_0*j[int(kstart):int(kend+1)]
        j[int(kstart):int(kend+1)] = (D[int(kstart):int(kend+1)] - nu[int(kstart):int(kend+1)]* \
                                      F[int(kstart):int(kend+1)] + (wp[int(kstart):int(kend+1)]**2)*\
                                      C[int(kstart):int(kend+1)]*F[int(kstart):int(kend+1)]/ \
                                      B[int(kstart):int(kend+1)])*j[int(kstart):int(kend+1)] + \
                                      (eps_0*(wp[int(kstart):int(kend+1)]**2)*C[int(kstart):int(kend+1)]/ \
                                       B[int(kstart):int(kend+1)])*ez[int(kstart):int(kend+1)] +\
                                      (eps_0*(wp[int(kstart):int(kend+1)]**2)*F[int(kstart):int(kend+1)] - \
                                       eps_0*(wp[int(kstart):int(kend+1)]**2)*C[int(kstart):int(kend+1)]* \
                                       D[int(kstart):int(kend+1)]/B[int(kstart):int(kend+1)])*ez_0[int(kstart):int(kend+1)]
        ez_0 = ez.copy() # Store current time step Ez field for next iteration
        
        # Spatial loop from trailing edge of plasma to end of domain Ez field
        # k=end+1
        # while k<=Kx:
        #     ez[k] = ez[k] + dt/eps_0/ddx*(hy[k]-hy[k-1])
        #     k+=1
        ez[kend:] = ez[kend:] + dt/eps_0/ddx*(hy[kend:]-hy[kend-1:-1])
        
        # Crude absorbing boundaries (hacked from a file on the Mathworks site)
        ez[1]=ez_low_m2
        ez_low_m2=ez_low_m1
        ez_low_m1=ez[2]
        ez[round(Kx)]=ez_high_m2
        ez_high_m2=ez_high_m1
        ez_high_m1=ez[round(Kx-1)]
        
        # Spatial loop for magnetic flux field, Hy
        # k=1
        # while k<=Kx-1:
        #     hy[k] = hy[k] + dt/mu_0/ddx*(ez[k+1]-ez[k])
        #     k+=1
        hy[1:-1] = hy[1:-1] + dt/mu_0/ddx*(ez[:-2]-ez[1:-1])
        
        # Probe to collect input signal in time domain
        #if n<ddx/c_0/dt*kstart # Time index set so I don't overlap input and reflected signals 
        if n<np.int64(.3/c_0/dt):
            geninput[n] = round(ez[int(probe_input)].real,4) # Watch the index if you change the spatial grid
        else:
            geninput[n] = 0
        # Probes to collect reflected and transmitted signals in time domain
        if n>np.int64(.3/c_0/dt):
            trans[n] = round(ez[int(probe_transmitted)].real,4)
            refl[n] = round(ez[int(probe_reflected)].real,4) # Watch the index if you change the spatial grid
        else:
            trans[n] = 0
            refl[n] = 0
        real = appending1D(ez) #geninput+refl+trans
        animatedPlot.append(real)
        
elif TestLoop == 1:
    for n in range(0,int(nsteps)): 
        if n%100 == 0:
            print("this is n %0.0f" %n)
        # Spatial loop up to front edge of plasma for Ez field
        k=2
        while k<=Kx:
            ez[k] = ez[k] + dt/eps_0/ddx*(hy[k]-hy[k-1])
            k+= 1
        # Source Pulse
        pulse = -amplitude*(t0-T)/spread*np.exp(-((t0-T)/spread)**2) # Derivative Gaussian input signal
        #pulse = 40*sin(2*pi*3e9*T) # Sine input with single frequency
        ez[10]=ez[10]+pulse
        T+=dt
        
        # Crude absorbing boundaries (hacked from a file on the Mathworks site)
        ez[1]=ez_low_m2
        ez_low_m2=ez_low_m1
        ez_low_m1=ez[2]
        ez[round(Kx)]=ez_high_m2
        ez_high_m2=ez_high_m1
        ez_high_m1=ez[round(Kx-1)]
        # Spatial loop for magnetic flux field, Hy
        k=1
        while k<=Kx-1:
            hy[k] = hy[k] + dt/mu_0/ddx*(ez[k+1]-ez[k])
            k+=1
        # Probe to collect input signal in time domain
        #if n<ddx/c_0/dt*kstart # Time index set so I don't overlap input and reflected signals 
        if n<np.int64(.3/c_0/dt):
            geninput[n] = round(ez[int(probe_input)].real,4) # Watch the index if you change the spatial grid
        else:
            geninput[n] = 0
        # Probes to collect reflected and transmitted signals in time domain
        if n>np.int64(.3/c_0/dt):
            trans[n] = round(ez[int(probe_transmitted)].real,4)
            refl[n] = round(ez[int(probe_reflected)].real,4) # Watch the index if you change the spatial grid
        else:
            trans[n] = 0
            refl[n] = 0
        real = appending1D(ez) #geninput+refl+trans
        animatedPlot.append(real)
elif TestLoop == 2:
    for n in range(0,int(nsteps)): 
        if n%100 == 0:
            print("this is n %0.0f" %n)
        # Spatial loop up to front edge of plasma for Ez field
        k=2
        while k<=int(kstart-1):
            ez[k] = ez[k] + dt/eps_0/ddx*(hy[k]-hy[k-1]);
            k+= 1
        # Source Pulse
        pulse = -amplitude*(t0-T)/spread*np.exp(-((t0-T)/spread)**2) # Derivative Gaussian input signal
        #pulse = 40*sin(2*pi*3e9*T) # Sine input with single frequency
        ez[10]=ez[10]+pulse
        T+=dt
        # Spatial loop through plasma with ME-FDTD formulation for Ez field
        k=int(kstart)
        end = int(kend)
        while k<=end:
            ez[k] = D[k]*ez[k] + B[k]/eps_0/ddx*(hy[k]-hy[k-1]) - F[k]/eps_0*j[k]
            j[k] = (D[k] - nu[k]*F[k] + (wp[k]**2)*C[k]*F[k]/B[k])*j[k] + (eps_0*(wp[k]**2)*C[k]/B[k])*ez[k] +\
                (eps_0*(wp[k]**2)*F[k] - eps_0*(wp[k]**2)*C[k]*D[k]/B[k])*ez_0[k]
            ez_0[k] = ez[k] # Store current time step Ez field for next iteration
            k+= 1
            if n >=400 and n<1000:
                print(f'This is j: {j[k]}')
        # Spatial loop from trailing edge of plasma to end of domain Ez field
        k=end+1
        while k<=Kx:
            ez[k] = ez[k] + dt/eps_0/ddx*(hy[k]-hy[k-1])
            k+=1
        # Crude absorbing boundaries (hacked from a file on the Mathworks site)
        ez[1]=ez_low_m2
        ez_low_m2=ez_low_m1
        ez_low_m1=ez[2]
        ez[round(Kx)]=ez_high_m2
        ez_high_m2=ez_high_m1
        ez_high_m1=ez[round(Kx-1)]
        
        # Spatial loop for magnetic flux field, Hy
        k=1
        while k<=Kx-1:
            hy[k] = hy[k] + dt/mu_0/ddx*(ez[k+1]-ez[k])
            k+=1
        # Probe to collect input signal in time domain
        #if n<ddx/c_0/dt*kstart # Time index set so I don't overlap input and reflected signals 
        if n<np.int64(.3/c_0/dt):
            geninput[n] = round(ez[int(probe_input)].real,4) # Watch the index if you change the spatial grid
        else:
            geninput[n] = 0
        # Probes to collect reflected and transmitted signals in time domain
        if n>np.int64(.3/c_0/dt):
            trans[n] = round(ez[int(probe_transmitted)].real,4)
            refl[n] = round(ez[int(probe_reflected)].real,4) # Watch the index if you change the spatial grid
        else:
            trans[n] = 0
            refl[n] = 0
        real = appending1D(ez) #geninput+refl+trans
        animatedPlot.append(real)
        
if Want2Animate == 1:
    # Plot signal. Animation help from 1D Electromagnetic FdTD in Python by natsunoyuki from GitHub
    figure, ax = plt.subplots()
    xx ,yy = [],[]
    line, = ax.plot([],[],'ok')
    ax.set_xlabel('Propagation axis (m)')
    ax.set_ylabel('Signal (V/m)')
    def init():
        ax.set(xlim=[0, L], ylim=[-50, 50])
        ax.vlines(kstart*ddx,-50, 50,color = 'm')
        ax.vlines(kend*ddx,-50, 50,color = 'm')
        ax.grid()
        return line,
    #text((.06),47,'Free Space'); 
    #text((.22),47,'Plasma','Color','m'); 
    #text((.36),47,'Free Space');
    #animation
    # 
    # 
    # INPUTS: 
    #   n :
    def animations(frames):
        if frames%100 == 0:
            print("this is the frame %0.0f" %frames)
        figure.suptitle('Time step = %0.0f' %frames)
        xx = np.linspace(0,L,int(Kx+1))
        yy = animatedPlot[frames]
        # i=0
        # if frames%100 == 0:
        #     while i < int(Kx+1):
        #         print("this is the y %0.3f" %yy[i])
        #         i+=1
        line.set_data(xx,yy)
        return line,
    anim = animation.FuncAnimation(fig = figure, func = animations, frames = range(0,int(nsteps),50), init_func=init,blit = True) #len(ez) - 1
    anim.save("Plasma_1Dfdtd_animation.gif", writer = "pillow")
    plt.show()
else:
    animatedPlot = []

  
#toc;
#runtime = toc;

#Close the video file containing the animation
#close(vidObj);
##
# Plot time domain signals
figure1, (p1,p2,p3) = plt.subplots(3,1,sharex = True)
#p1.subplot(3,1,1)
p1.plot(t,geninput) 
p1.set_xlabel('Time step')
p1.set_ylabel('Input\n(V/m)')
#title({['Time = ',time_stamp,' sec'];' '});
#p2.subplot(3,1,2)
p2.plot(t,refl)
p2.set_xlabel('Time step')
p2.set_ylabel('Reflected\n(V/m)')
#p3.subplot(3,1,3)
p3.plot(t,trans)
p3.set_xlabel('Time step')
p3.set_ylabel('Transmitted\n(V/m)')
figure1.suptitle('Time Domian Signals')
plt.show()

# Sampling frequency
Fs=1/dt
# Fourier transform of signals to the frequency domain
#Y1 = np.fft.fft(geninput)
#Y2 = np.fft.fft(trans)
#Y3 = np.fft.fft(refl)
Y1 = fft(geninput)
Y2 = fft(trans)
Y3 = fft(refl)
f = Fs*np.linspace(0,1,len(Y1)) # frequency variable
# Plot spectrum of input signal
Py1 = Y1*np.conj(Y1)
figure2, (plt1) = plt.subplots(1,1)
plt1.plot(f/(10**9),abs(Py1))
plt1.set_xlim(0, 11) 
plt1.set_ylim(0, 1.2*max(abs(Py1)))
plt1.set_xlabel('Frequency (GHz)')
plt1.set_ylabel('PSD of Input Signal')
figure2.suptitle('Derivative Gaussian Pulse Distribution')
plt1.vlines([0.39, 1.550, 5.2], 0, 1.2*max(abs(Py1)), color = 'm', label = ['L-Band','S-Band','X-Band'])
plt1.vlines([3.9, 6.2], 0, 1.2*max(abs(Py1)),color = 'b', label = ['C-Band','C-Band'])
plt1.grid()
plt.show()
# Calculate Transmission and Reflection coefficients (loss in dB)
magT1 = []
magR = []
T1 = Y2/Y1
R = Y3/Y1
#print("this is the length of t %0.0f" %len(t))
#for nsteps = 40,000 T1*T1 become far to large after 8000 when the range went to t
# for nsteps = 40000 T1*T1 become far to large after 10000 when the range went to T1
for  v in range(0,len(T1)): #t
    #print("this is v %0.0f" %v)
    magT1.append(10*cmath.log10(abs(T1[v]*T1[v])))
    magR.append(10*cmath.log10(abs(R[v]*R[v])))
    
##
d = (kend-kstart)*ddx
# Plot reflected and transmission coefficients
#scrsz = get(0,'ScreenSize');
#figure('OuterPosition',[scrsz(3)/2 scrsz(4)/4.5 scrsz(3)/2 scrsz(4)/1.5])
#plt.rcParams["figure.autolayout"] = True
figure3, (ax1, ax2) = plt.subplots(2,1)
ax1.set_xlim(0, 10)
ax1.set_ylim(1.2*min(np.real(magR)), 1)
ax1.grid()
ax1.set_xlabel("Frequency (GHz)")
ax1.set_ylabel("Reflection coefficient\n magnitude (dB)")
ax1.vlines([0.39, 1.550, 5.2],1.2*min(np.real(magR)), 1,color = 'm', label = ['L-Band','S-Band','X-Band'])
ax1.vlines([3.9, 6.2],1.2*min(np.real(magR)), 1,color = 'b', label = ['C-Band','C-Band'])
ax1.plot(f/(10**9),magR,'o'or ' ', c = 'k' ,ms = 3) #(range(1,len(f)))
wHz_exact_correct = [0]*len(wHz_exact)
for i in range(0,len(wHz_exact)):
    wHz_exact_correct[i] = wHz_exact[i]/(10**9)
ax1.plot(wHz_exact_correct,magR_exact,c = 'r') #'#8AF4FF'
#------------FIGURE OUT WHICH GOES WHERE DURING DEBUGGING---------------------#
#ax1.legend(['ME-FDTD','Exact','Location','SouthEast'])

ax2.set_ylim(1.2*min(np.real(magT1)), 1)
ax2.set_xlim(0, 10)
ax2.grid()
ax2.set_xlabel("Frequency (GHz)")
ax2.set_ylabel('Transmission coefficient\n magnitude (dB)')
ax2.vlines([0.39, 1.550, 5.2],1.2*min(np.real(magT1)), 1,color = 'm', label = ['L-Band','S-Band','X-Band'])
ax2.vlines([3.9, 6.2],1.2*min(np.real(magT1)), 1,color = 'b', label = ['C-Band','C-Band'])
ax2.plot(f/(10**9),magT1,'o'or ' ', c = 'k' ,ms = 3)
ax2.plot(wHz_exact_correct,magT_exact,c = 'r')#'#8AF4FF'
#------------FIGURE OUT WHICH GOES WHERE DURING DEBUGGING---------------------#
#ax2.legend(['ME-FDTD','Exact','Location','SouthEast'])
figure3.suptitle('Reflected and Transmission Coefficients')

plt.show()

##
# Interpolate attenuation at selected frequencies and output to a file
#output(:,1)=stamp;
#for n = 1:length(fxmt)
#    Txmt(n) = interp1(f,magT1,fxmt(n),'chip'); # Interpolated Transmission coefficient in dB
#    output(:,n+1) = Txmt(n);
#output(:,length(fxmt)+2) = ddx;
#output(:,length(fxmt)+3) = dt;
#output(:,length(fxmt)+4) = nsteps;
#output(:,length(fxmt)+5) = max(wpHz);
#output(:,length(fxmt)+6) = max(nu);
#output(:,length(fxmt)+7) = d;
#output(:,length(fxmt)+8) = toc;
#save(filename_out, 'output', '-ASCII', '-append')
#clear output

    
    
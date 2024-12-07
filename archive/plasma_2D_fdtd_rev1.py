# ******
# Two-dimensional Matrix Exponential Finite Difference Time Domain 
# (ME-FDTD) electromagnetic propagation code
# Revision 1:
#    5 June 2024. Meredith Amrhein 
#      Created the base of the code, moving over 1D to 2D.
#    2 July 2024. Cody Parden  
#      Created the code for calculating the Power and Energy and  
#      Graphing them
# Revision 2:
#    11 Juy 2024. Meredith Amrhein (Started to make the code faster)
# The purpose of this 2D ME-FDTD EM code is to calculate the attenuation
# through a spatially varying plasma sheath.  The code is set up to read 
# in .txt (US3D) output with the plasma frequency and collision frequency given
# along a propagation axis.  The code calculates the reflection and
# transmission coefficients for the electric field caused by the plasma
# field.  The transmission coefficient in turn can be used to calculate the
# signal loss from attenuation in the plasma field.
 
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
#Libaries
import math
import cmath
from Plasma_FDTD_Functions_rev1 import *

from scipy.interpolate import interp1d
from scipy.fft import fft, ifft
from mpl_toolkits import mplot3d

import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation 
from mpl_toolkits.axes_grid1 import make_axes_locatable
import textwrap

Dimention = 2
#-----------------Variables to change----------------------
#Animation need to be generated as a pcolormesh plot: 2
#Animation need to be generated as a 3D surface plot: 1
#Animation not need to be generated                 : 0
Want2Animate = 0
#Does Hy need to be animated?
plotH = False
#hx or hy
letter ="y"
#Does EZ need to be animated
plotEz = True
#Is there no data: True
#Is there data   : False
NOdata = True

#Any graphs you do not want set the value at that index to 0
#Index 0: Plasma and Collision Frequency in 2D graphing x and y 
#Index 1: Plasma and Collision Frequency graphed in 3D on two different graphs
#Index 2: Plasma and Collision Frequency graphed in 3D on one graph 
#                               (cannot see the collision)
#Index 3: Plasma and Collision Frequency graphed on a pcolormesh
#Index 4: Input, Reflected and Transmitted graphs
#Index 5: Input Zoomed, Input, pulse, pulse zoomed graphs
#Index 6: Derivative Gaussian Pulse Distribution
#Index 7: Reflected and Transmitted Coefficients
#Index 8: Power/ Energy Density Graph
Figures2Print = [0,0,0,0,1,0,1,1,1,1]

#What direction/ kind of input into the plasma
#make all other indexs 0
#Index 0: Point Source or Ossilating Line: Plasma (when the plasma is only changing in on direction but fills the other direction)
#Index 1: Point Source or Ossilating Line: Plasma (kstart-kbottom must be above 2 and kend-ktop must be smaller than Kx-y-2)
#Index 2: Point Source or Ossilating Line: no plasma
#Index 3: Point Source or Ossilating Line: Plasma (Only in X direction) BEST ONE!!!!!!!
typeofinput = [0,0,0,1]

# This variable decides the probing points for the inpput, reflected and transmitted points
# "x" if the plasma is only changing in the x direction (ktop and kbottom are 0)
# "xy" if there is plasma in 2 directions (kbttom and ktop are not zero)
# "y" if the plasma is only changing in the y direction (kstart and kend are 0) NOT READY
PlasmaDirection = "x"
#What file has all the data that needs to be passed through the simulation
filename = "Data1D"
#-----------------------end--------------------------------
#Freaking AmazingPlasma Animation Numbers Lx:0.8 Ly:0.8 ddx:0.0005 or ddx:0.008
Lx=0.1                    # Length of the x-portion of the EM domain in meters 0.4
Ly=0.1                    # Length of the y-portion of the EM domain in meters 0.08
ddx = 0.0001              # Delta-X step, meters 0.001 (For 0.0025 need to have 30000 n-steps)
ddy=(Ly*ddx)/Lx           # Delta-Y step, meters
dd = [ddx,ddy,0]
Kx=int(Lx/ddx)            # Number of spatial grid points in the x-direction
Ky=int(Ly/ddy)            # Number of spatial grid points in the y-direction
KE = [Kx,Ky,0]
nsteps=1500              # Number of time steps to run 
eps_0=8.854*(10**-12)     # free space permittivity  
mu_0=4*math.pi*(10**-7)   # free space permeability
c_0 = 2.9975*(10**8)      # free space wave speed in m/s
CFL = 0.5                 # Courant-Friedrichs-Lewy number
dt = CFL*min(ddx,ddy)/c_0 # Time step   
kstart= int(0.1/ddx)      # Front edge index of plasma sheath (x)
kend= int(0.15/ddx)        # Back edge index of plasma sheath  (x)   try this later:(0.099/ddx) 
kbottom = int(0.001/ddy)  # Bottom index for plasma sheath    (y)
ktop = int(0.199/ddy)     # Top index for plasma sheath       (y)
kleft = 0                 # left index for plasma sheath      (z)
kright = 0                # right index for plasma sheath     (z)
plasmaStart = [kstart,kbottom,kleft]
plasmaEnd = [kend,ktop,kright]
q = -1.6*(10**-19)
#Spatial grids for plotting
x= []    
for i in range(0,Kx):
    x.append(i*ddx)
y= []
for i in range(0,Ky):
    y.append(i*ddy)

#Setting all inital conditions to zero
ez=np.outer([0]*(Kx+1),np.ones(Ky+1))       # Initial electric field (Current Time step)
ez_0=np.outer([0]*(Kx+1),np.ones(Ky+1))     # Storage electric field variable for ME-FDTD
ez_n_1 = np.outer([0]*(Kx+1),np.ones(Ky+1)) # Storage electric field variable for Bondary Conditions (-2 previous time step)
ez_n = np.outer([0]*(Kx+1),np.ones(Ky+1))   # Storage electric field variable for Bondary Conditions (-1 previous time step)
j=np.outer([0]*(Kx+1),np.ones(Ky+1))        # Initial current density
hx=np.outer([0]*(Kx+1),np.ones(Ky))         # Initial x magnetic flux density
hy=np.outer([0]*Kx,np.ones(Ky+1))           # Initial y magnetic flux density
HH =np.outer([0]*(Kx+1),np.ones(Ky+1))      # Magnetic field variable used for animating

pulseArray = [0]*nsteps                          # Pulse variable (used to double check that the pulse is correct)
geninput=[0]*nsteps                              # Input probed at point 5
trans=[0]*nsteps                                 # Transmission probed 4 indexs after the plasma ends
refl=[0]*nsteps                                  # Reflection probed after the input and before the plasma
Time=0                                           # Inital Time to be used for the input wave (pulse) 

#Input wave parameters for derivative gaussian input
spread = 110*dt
t0 = 600*dt
amplitude = 100 

t = []
for i in range(0,nsteps):
    t.append(i*dt)
i=0

# Probe point variables (All are set in the individual typesofinput before main loops start)
probe_input = 0
probe_reflectedx = 0
probe_transmittedx = 0
probe_reflectedy = 0
probe_transmittedy = 0

# Plasma and Collision Frequency     
wpHz_plasma = 4*(10**9)
nuHz_plasma = 1*(10**9)

# More variables to be declared but it does not matter what the starting number is
# Most of these are used in the exact() function
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

#--------------------------------------------------------------------
#Processing data

#Functions in another file comFun_plasma.py
data = []
listOfColumns = []
if NOdata == False:
    listOfColumns,data = dataHandling(filename)
else:
    #faking data values while debugging because these will be replaced later
    #with the wpHz_plasma and nu_plasma
    listOfColumns = colNumList = [0,1,2,3,4]
    data = [[1]*5]*30

# If no data is present than this function will change to the plasma and collision frequency
# If there is data it will use that data over the declared plasma and collision frequency variables
wpHzx, nux, xint, wpHzy, nuy, yint, wpHzz, nuz, zint = WpNuSetup(Dimention,NOdata,data, listOfColumns, wpHz_plasma, nuHz_plasma, plasmaStart,plasmaEnd,KE,dd)

#This graph was to check all x and y values for the nu amd wpHz were collected correctly
#Used for debugging purposes 
if Figures2Print[0] == 1:    
    # Plot the plasma input data in 2D
    plt.semilogy(xint,wpHzx, color = 'r',label = 'Plasmax')
    plt.semilogy(yint,wpHzy, color = 'g', label = 'Plasmay')
    plt.semilogy(xint,nux, color = 'b', label = 'Collisionx')
    plt.semilogy(yint,nuy, color = 'k', label = 'Collisiony')
    plt.xlabel("Propagation distance (m)")
    plt.ylabel("Frequency (Hz)")
    plt.legend()
    plt.grid()
    plt.semilogy(kstart*ddx,kbottom*ddy, color = 'r', marker = 'o')
    plt.semilogy(kend*ddx,ktop*ddy, color = 'b', marker = 'o')
    #plt.imsave("3D_wpHz_nu.png",fig)
    plt.show()

# Below takes the x and y values calculated from the WpNuSetup and makes mesh grids of 
# wp and nu and creates meshgrid for the x and y values to be used to graph 
indexnum = 0
for i in range(0,len(wpHzx)-1):
    if (wpHzx[i-1] == 1) and (wpHzx[i] != 1) and (wpHzx[i+1] != 1):
        indexnum = i

xgraph = np.outer(np.linspace(0,Lx,Kx+1),np.ones(Kx+1))
xgraph = xgraph.copy().T
ygraph = np.outer(np.linspace(0,Ly,Ky+1),np.ones(Ky+1))
wpHzgraph = xgraph.copy()
nugraph = xgraph.copy()
for i in range(0,len(xint)):
    for k in range(0,len(yint)):
        if (xint[i] >= (kstart*ddx)) and (xint[i] <= (kend*ddx)) and (yint[k] >= (kbottom*ddy)) and (yint[k] <= (ktop*ddy)):
            wpHzgraph[i][k] = wpHzx[indexnum]
            nugraph[i][k] = nux[indexnum]
        else:
            wpHzgraph[i][k] = 1
            nugraph[i][k] = 1

# This graph makes a 3d surface plot of the wpHz and nu in 2 separte plots
if Figures2Print[1] == 1:
    fig = plt.figure(figsize = [12,13])
    ax = fig.add_subplot(2,1,1,projection='3d')
    one = ax.plot_surface(xgraph,ygraph,wpHzgraph,cmap='plasma',label = "Plasma")
    # ax.scatter([kstart*ddx],[kbottom*ddy],[0], color='r', marker = 'o')
    # ax.scatter([kend*ddx],[ktop*ddy],[0], color='b', marker = 'o')
    ax.set_xticks(np.arange(0,Lx+0.1, Lx/5))
    ax.set_xlabel("Propagation distance (m)")
    ax.set_yticks(np.arange(0,Ly+0.1,Ly/5))
    ax.set_ylabel("Propagation distance (m)")
    ax.set_zlim(-1,wpHzx[indexnum]+100)
    ax.set_zlabel("Frequency (Hz)")
    ax.set_title('Plasma Frequency')
    fig.colorbar(one, shrink=0.5, aspect=10)
    ax = fig.add_subplot(2,1,2,projection='3d')
    two = ax.plot_surface(xgraph,ygraph,nugraph,cmap='plasma',label = "Collision")
    ax.set_xticks(np.arange(0,Lx+0.1, Lx/5))
    ax.set_xlabel("Propagation distance (m)")
    ax.set_yticks(np.arange(0,Ly+0.1, Ly/5))
    ax.set_ylabel("Propagation distance (m)")
    ax.set_zlim(-1,nux[indexnum]+100)
    ax.set_zlabel("Frequency (Hz)")
    ax.set_title('Collision Frequency')
    fig.colorbar(two, shrink=0.5, aspect=10)
    plt.show()
# This graph combines the previous 2 plots. You cannot see the nu graph so I do not sugest using it
if Figures2Print[2] == 1:
    fig = plt.figure(figsize = [8,10])
    ax = plt.axes(projection='3d')
    one = ax.plot_surface(xgraph,ygraph,wpHzgraph,cmap='YlOrBr',label = "Plasma")
    two = ax.plot_surface(xgraph,ygraph,nugraph,cmap='YlGnBu',label = "Collision")
    ax.set_xticks(np.arange(0,Lx,Lx/5))
    ax.set_xlabel("Propagation distance (m)")
    ax.set_yticks(np.arange(0,Ly,Ly/6))
    ax.set_ylabel("Propagation distance (m)")
    ax.set_zlim(-1,wpHzx[indexnum]+100)
    ax.set_zlabel("Frequency (Hz)")
    ax.set_title('Plasma & Collision Frequency')
    fig.colorbar(one, shrink=0.5, aspect=10)
    fig.colorbar(two, shrink=0.5, aspect=10)
    plt.show()
# This graph has the plasma and collision Frequency graphed on a pcolormesh. It does not look as good as the surface 
# plot so I do not suggest using it
if Figures2Print[3] ==1:
    fig = plt.figure(figsize = [12,13])
    ax = fig.add_subplot(2,1,1)
    one = ax.pcolormesh(xgraph,ygraph,wpHzgraph,cmap='plasma',label = "Plasma")
    # ax.scatter([kstart*ddx],[kbottom*ddy],[0], color='r', marker = 'o')
    # ax.scatter([kend*ddx],[ktop*ddy],[0], color='b', marker = 'o')
    ax.set_xticks(np.arange(0,Lx+0.1, Lx/5))
    ax.set_xlabel("Propagation distance (m)")
    ax.set_yticks(np.arange(0,Ly+0.1,Ly/5))
    ax.set_ylabel("Propagation distance (m)")
    # ax.set_zlabel("Frequency (Hz)")
    ax.set_title('Plasma Frequency')
    fig.colorbar(one, shrink=0.5, aspect=10)
    ax = fig.add_subplot(2,1,2)
    two = ax.pcolormesh(xgraph,ygraph,nugraph,cmap='plasma',label = "Collision")
    ax.set_xticks(np.arange(0,Lx+0.1, Lx/5))
    ax.set_xlabel("Propagation distance (m)")
    ax.set_yticks(np.arange(0,Ly+0.1,Ly/5))
    ax.set_ylabel("Propagation distance (m)")
    # ax.set_zlabel("Frequency (Hz)")
    ax.set_title('Collision Frequency')
    fig.colorbar(two, shrink=0.5, aspect=10)
    plt.show()
    
# Convert plasma frequency to angular frequency (and move nugraph label to label nu for simplicity)
wp = wpHzgraph.copy()
wp = 2*math.pi*wpHzgraph
nu = nugraph.copy()   

#Below calculates the exact values for the wHz, magT and magR  
# (magnitude of Transmission and Reflection Coeficients)
wHz_exactx , magR_exactx, magT_exactx = exact(kstart,kend,ddx,c_0,wpHz_plasma,nuHz_plasma)
wHz_exacty , magR_exacty, magT_exacty = exact(kbottom,ktop,ddy,c_0,wpHz_plasma,nuHz_plasma)

# ME-FDTD parameters
aa=[]
bb = []
b= []
D = []
F = []
B = []
C = [] 
a = []
DD = []
FF = []
BB = []
CC = []
# D= np.outer([0]*(Kx+1),np.ones(Ky+1)) 
# F= np.outer([0]*(Kx+1),np.ones(Ky+1)) 
# B= np.outer([0]*(Kx+1),np.ones(Ky+1)) 
# C= np.outer([0]*(Kx+1),np.ones(Ky+1)) 

#VIDEO VARIABLES
animatedPlot = []
animatedPlotH = []

######Main Loop Begins######
#tic; 
# Outer time step loop
if typeofinput[0] == 1:
   # ME-FDTD parameters
   a= np.outer([0]*(Kx+1),np.ones(Ky+1))
   b= np.outer([0]*(Kx+1),np.ones(Ky+1)) 
   D= np.outer([0]*(Kx+1),np.ones(Ky+1)) 
   F= np.outer([0]*(Kx+1),np.ones(Ky+1)) 
   B= np.outer([0]*(Kx+1),np.ones(Ky+1)) 
   C= np.outer([0]*(Kx+1),np.ones(Ky+1)) 
   # Calculate ME-FDTD parameters
   a = np.power((np.power(nu,2,dtype = complex))-(4*np.power(wp,2,dtype = complex)),0.5)/2 # This piece of eigenvalue can be complex ...
   b = (nu/2)
   D = (a+b)/(2*a)*np.exp((a-b)*dt) + ((a-b)/(2*a)*np.exp((-a-b)*dt))
   F = np.exp((a-b)*dt) - np.exp((-a-b)*dt)/(2*a)
   B = (a+b)/(2*a*(a-b))*(np.exp((a-b)*dt)-1) - (a-b)/(2*a*(a+b))*(np.exp((-a-b)*dt)-1)
   C = 1/(2*a*(a-b))*(np.exp((a-b)*dt)-1) + 1/(2*a*(a+b))*(np.exp((-a-b)*dt)-1)
   
   probe_input = 10
   probe_reflectedx = int(Kx/2)
   probe_transmittedx = kend + 5
   probe_reflectedy = iny(Ky/2)
   probe_transmittedy = probe_input
   
   
   for n in range(0,nsteps): 
       if n%100 == 0:
           print("this is n %0.0f" %n)  
       if (kbottom == (0 or 1 or 2)) and (ktop == (Ky or (Ky-1) or (Ky-2))):
           
           #Before the plasma      
           ez[1:kstart-1,1:-1] = ez[1:kstart-1,1:-1] +dt/eps_0 *\
               ((hy[1:kstart-1,1:-1]-hy[:kstart-2,1:-1])/ddx - \
                (hx[1:kstart-1,1:]-hx[1:kstart-1,:-1])/ddy)
           
           # Source Pulse
           if n <= 750:
               pulse = -amplitude*(t0-Time)/spread*np.exp(-((t0-Time)/spread)**2) # Derivative Gaussian input signal
               # pulse = 40*np.sin(2*np.pi*3e9*Time) # Sine input with single frequency
               # pulse = 10
           else: 
               pulse = 0
           pulseArray[n] = pulse
           ez[10,:]=ez[10,:]+pulse
           Time+=dt
           
           # Through plasma with ME-FDTD formulation for Ez field
           #OLD
           ez[kstart:kend,1:-1] = \
                D[kstart:kend,1:-1].real*ez[kstart:kend,1:-1] + \
                    B[kstart:kend,1:-1].real/eps_0 * \
                        ((hy[kstart:kend,1:-1]-\
                          hy[kstart-1:kend-1,1:-1])/ddx - \
                         (hx[kstart:kend,1:]- \
                          hx[kstart:kend,:-1])/ddy) - \
                            F[kstart:kend,1:-1].real/eps_0*j[kstart:kend,1:-1]
           # ez[kstart:kend,1:-1] = \
           #     D[kstart:kend,1:-1].real*ez[kstart:kend,1:-1] + \
           #         B[kstart:kend,1:-1].real/eps_0 * \
           #             ((hy[kstart:kend,1:-1]-\
           #               hy[kstart-1:kend-1,1:-1])/ddx - \
           #              (hx[kstart:kend,1:]- \
           #               hx[kstart:kend,:-1])/ddy) - \
           #                 F[kstart:kend,1:-1].real/eps_0*((jy[kstart:kend,1:-1]-jy[kstart-1:kend-1,1:-1]) \
           #                                                           -jx[kstart:kend,1:]-jx[kstart:kend,:-1])
           # j = (D.real - nu*F.real + (wp**2)*C.real*F.real/B.real)*j + eps_0*(wp**2)*C.real/B.real*ez + (eps_0*(wp**2)*F.real - eps_0*(wp**2)*C.real*D.real/B.real)*ez_0
           #OLD
           j[kstart:kend,1:-1] = (D[kstart:kend,1:-1].real - \
                nu[kstart:kend,1:-1]*F[kstart:kend,1:-1].real + \
                    (wp[kstart:kend,1:-1]**2)*C[kstart:kend,1:-1].real* \
                        F[kstart:kend,1:-1].real/B[kstart:kend,1:-1].real)*\
                          j[kstart:kend,1:-1] + eps_0*(wp[kstart:kend,1:-1]**2)*\
                              C[kstart:kend,1:-1].real/B[kstart:kend,1:-1].real*\
                                  ez_0[kstart:kend,1:-1] + (eps_0*(wp[kstart:kend,1:-1]**2)*\
                                      F[kstart:kend,1:-1].real - eps_0*(wp[kstart:kend,1:-1]**2)*\
                                          C[kstart:kend,1:-1].real*D[kstart:kend,1:-1].real/ \
                                              B[kstart:kend,1:-1].real)*ez_0[kstart:kend,1:-1]
           
           # jx[kstart:kend,1:] = (D[kstart:kend,1:-1].real - \
           #     nu[kstart:kend,1:-1]*F[kstart:kend,1:-1].real + \
           #         (wp[kstart:kend,1:-1]**2)*C[kstart:kend,1:-1].real* \
           #             F[kstart:kend,1:-1].real/B[kstart:kend,1:-1].real)*\
           #               jx[kstart:kend,1:] + eps_0*(wp[kstart:kend,1:-1]**2)*\
           #                   C[kstart:kend,1:-1].real/B[kstart:kend,1:-1].real*\
           #                       ez[kstart:kend,1:-1] + (eps_0*(wp[kstart:kend,1:-1]**2)*\
           #                           F[kstart:kend,1:-1].real - eps_0*(wp[kstart:kend,1:-1]**2)*\
           #                               C[kstart:kend,1:-1].real*D[kstart:kend,1:-1].real/ \
           #                                   B[kstart:kend,1:-1].real)*(ez_0[kstart:kend,1:]-ez_0[kstart:kend,:-1])
           
           # jy[kstart:kend,1:-1] = (D[kstart:kend,1:-1].real - \
           #     nu[kstart:kend,1:-1]*F[kstart:kend,1:-1].real + \
           #         (wp[kstart:kend,1:-1]**2)*C[kstart:kend,1:-1].real* \
           #             F[kstart:kend,1:-1].real/B[kstart:kend,1:-1].real)*\
           #               jy[kstart:kend,1:-1] + eps_0*(wp[kstart:kend,1:-1]**2)*\
           #                   C[kstart:kend,1:-1].real/B[kstart:kend,1:-1].real*\
           #                       ez[kstart:kend,1:-1] + (eps_0*(wp[kstart:kend,1:-1]**2)*\
           #                           F[kstart:kend,1:-1].real - eps_0*(wp[kstart:kend,1:-1]**2)*\
           #                               C[kstart:kend,1:-1].real*D[kstart:kend,1:-1].real/ \
           #                                   B[kstart:kend,1:-1].real)*(ez_0[kstart:kend,1:-1]-ez_0[kstart-1:kend-1,1:-1])
           ez_0 = ez # Store current time step Ez field for next iteration 
           
           #After the plasma  
           ez[kend:-1,1:-1] = ez[kend:-1,1:-1] +dt/eps_0 * \
               ((hy[kend:,1:-1]-\
                 hy[kend-1:-1,1:-1])/ddx - \
                (hx[kend:-1,1:]-\
                 hx[kend:-1,:-1])/ddy)
           
           #Mur Bondary Conditions from book 1st and second order
           #Left
           #1st
           ez[0,:] = ez_n[1,:] +((c_0*dt-ddx)/(c_0*dt+ddx))*(ez[1,:]-ez_n[0,:])
           #2nd
           # ez[0,1:-1] = -ez_n_1[1,1:-1]-((ddx-(c_0*dt))/(ddx+(c_0*dt)))*(ez[1,1:-1]+ez_n_1[0,1:-1]) \
           #     + (2*ddx/(ddx+(c_0*dt)))*(ez_n[0,1:-1]+ez_n[1,1:-1]) \
           #     + ((ddx*(c_0*dt)**2)/(2*(ddy**2)*(ddx+c_0*dt)))*\
           #         (ez_n[0,2:]-2*ez_n[0,1:-1]+ez_n[0,:-2]+\
           #           ez_n[1,2:]-2*ez_n[1,1:-1]+ez_n[1,:-2])
           #Bottom
           #1st
           ez[:,0] = ez_n[:,1] +((c_0*dt-ddy)/(c_0*dt+ddy))*(ez[:,1]-ez_n[:,0])
           #2nd
           # ez[1:-1,0] = -ez_n_1[1:-1,1]-((ddx-(c_0*dt))/(ddx+(c_0*dt)))*(ez[1:-1,1]+ez_n_1[1:-1,0]) \
           #     + (2*ddx/(ddx+(c_0*dt)))*(ez_n[1:-1,0]+ez_n[1:-1,1]) \
           #     + ((ddx*(c_0*dt)**2)/(2*(ddy**2)*(ddx+c_0*dt)))*\
           #         (ez_n[2:,0]-2*ez_n[1:-1,0]+ez_n[:-2,0]+\
           #           ez_n[2:,1]-2*ez_n[1:-1,1]+ez_n[:-2,1])
           #Right
           #1st
           ez[Kx,:]=ez_n[Kx-1,:] +((c_0*dt-ddx)/(c_0*dt+ddx))*(ez[Kx-1,:]-ez_n[Kx,:])
           #2nd
           # ez[Kx,1:-1] = -ez_n_1[Kx-1,1:-1]-((ddx-(c_0*dt))/(ddx+(c_0*dt)))*(ez[Kx-1,1:-1]+ez_n_1[Kx,1:-1]) \
           #     + (2*ddx/(ddx+(c_0*dt)))*(ez_n[Kx,1:-1]+ez_n[Kx-1,1:-1]) \
           #     + ((ddx*(c_0*dt)**2)/(2*(ddy**2)*(ddx+c_0*dt)))*\
           #         (ez_n[Kx,2:]-2*ez_n[Kx,1:-1]+ez_n[Kx,:-2]+\
           #           ez_n[Kx-1,2:]-2*ez_n[Kx-1,1:-1]+ez_n[Kx-1,:-2])
           #Top
           #1st
           ez[:,Ky]=ez_n[:,Ky-1] +((c_0*dt-ddy)/(c_0*dt+ddy))*(ez[:,Ky-1]-ez_n[:,Ky])
           #2nd
           # ez[1:-1,Ky] = -ez_n_1[1:-1,Ky-1]-((ddx-(c_0*dt))/(ddx+(c_0*dt)))*(ez[1:-1,Ky-1]+ez_n_1[1:-1,Ky]) \
           #     + (2*ddx/(ddx+(c_0*dt)))*(ez_n[1:-1,Ky]+ez_n[1:-1,Ky-1]) \
           #     + ((ddx*(c_0*dt)**2)/(2*(ddy**2)*(ddx+c_0*dt)))*\
           #         (ez_n[2:,Ky]-2*ez_n[1:-1,Ky]+ez_n[:-2,Ky]+\
           #           ez_n[2:,Ky-1]-2*ez_n[1:-1,Ky-1]+ez_n[:-2,Ky-1])
           
           hy = hy + (dt/(mu_0*ddx))*(ez[1:,:]-ez[:-1,:])
           hx = hx - (dt/(mu_0*ddy))*(ez[:,1:]-ez[:,:-1])
           
       elif (kstart == (0 or 1 or 2)) and (kend == (Kx or (Kx-1) or (Kx-2))):
           #Before the plasma   
           ez[1:-1,1:kbottom-1] = ez[1:-1,1:kbottom-1] +dt/eps_0 * ((hy[1:,1:kbottom-1]-hy[:-1,1:kbottom-1])/ddx - \
                                                        (hx[1:-1,1:kbottom-1]-hx[1:-1,:int(kbottom-2)])/ddy)
        
           # # Source Pulse
           # if n >= 5:
           #     # pulse = -amplitude*(t0-Time)/spread*np.exp(-((t0-Time)/spread)**2) # Derivative Gaussian input signal
           #     # pulse = 40*np.sin(2*np.pi*3e9*Time) # Sine input with single frequency
           #     # pulse = 10
           # else: 
           #     pulse = 0
           pulse = -amplitude*(t0-Time)/spread*np.exp(-((t0-Time)/spread)**2) # Derivative Gaussian input signal    
           pulseArray[n] = pulse
           ez[10,:int(0.4/ddy)]=ez[10,:int(0.4/ddy)]+pulse
           Time+=dt
           
           # Through plasma with ME-FDTD formulation for Ez field   
           ez[1:-1,kbottom+1:ktop-1] = \
               D[1:-1,kbottom+1:ktop-1].real*ez[1:-1,kbottom+1:ktop-1] + \
                   B[1:-1,kbottom+1:ktop-1].real/eps_0 * \
                       ((hy[1:,kbottom+1:ktop-1]-\
                         hy[:-1,kbottom+1:ktop-1])/ddx - \
                        (hx[1:-1,kbottom+1:ktop-1]- \
                         hx[1:-1,kbottom:int(ktop-2)])/ddy) - \
                           F[1:-1,kbottom+1:ktop-1].real/eps_0*j[1:-1,kbottom+1:ktop-1]
         
           # j = (D.real - nu*F.real + (wp**2)*C.real*F.real/B.real)*j + eps_0*(wp**2)*C.real/B.real*ez + (eps_0*(wp**2)*F.real - eps_0*(wp**2)*C.real*D.real/B.real)*ez_0
    
           j[1:-1,kbottom+1:ktop-1] = (D[1:-1,kbottom+1:ktop-1].real - \
               nu[1:-1,kbottom+1:ktop-1]*F[1:-1,kbottom+1:ktop-1].real + \
                   (wp[1:-1,kbottom+1:ktop-1]**2)*C[1:-1,kbottom+1:ktop-1].real* \
                       F[1:-1,kbottom+1:ktop-1].real/B[1:-1,kbottom+1:ktop-1].real)*\
                         j[1:-1,kbottom+1:ktop-1] + eps_0*(wp[1:-1,kbottom+1:ktop-1]**2)*\
                             C[1:-1,kbottom+1:ktop-1].real/B[1:-1,kbottom+1:ktop-1].real*\
                                 ez[1:-1,kbottom+1:ktop-1] + (eps_0*(wp[1:-1,kbottom+1:ktop-1]**2)*\
                                     F[1:-1,kbottom+1:ktop-1].real - eps_0*(wp[1:-1,kbottom+1:ktop-1]**2)*\
                                         C[1:-1,kbottom+1:ktop-1].real*D[1:-1,kbottom+1:ktop-1].real/ \
                                             B[1:-1,kbottom+1:ktop-1].real)*ez_0[1:-1,kbottom+1:ktop-1]
           ez_0 = ez # Store current time step Ez field for next iteration   
           
           #After the plasma  
           ez[kend+1:-1,ktop+1:-1] = ez[kend+1:-1,ktop+1:-1] +dt/eps_0 * \
               ((hy[kend+1:,ktop+1:-1]-\
                 hy[kend:-1,ktop+1:-1])/ddx - \
                (hx[kend+1:-1,ktop+1:]-\
                 hx[kend+1:-1,ktop:-1])/ddy)
           
           #Mur Bondary Conditions from book 1st and second order
           #Left
           #1st
           ez[0,:] = ez_n[1,:] +((c_0*dt-ddx)/(c_0*dt+ddx))*(ez[1,:]-ez_n[0,:])
           #2nd
           # ez[0,1:-1] = -ez_n_1[1,1:-1]-((ddx-(c_0*dt))/(ddx+(c_0*dt)))*(ez[1,1:-1]+ez_n_1[0,1:-1]) \
           #     + (2*ddx/(ddx+(c_0*dt)))*(ez_n[0,1:-1]+ez_n[1,1:-1]) \
           #     + ((ddx*(c_0*dt)**2)/(2*(ddy**2)*(ddx+c_0*dt)))*\
           #         (ez_n[0,2:]-2*ez_n[0,1:-1]+ez_n[0,:-2]+\
           #           ez_n[1,2:]-2*ez_n[1,1:-1]+ez_n[1,:-2])
           #Bottom
           #1st
           ez[:,0] = ez_n[:,1] +((c_0*dt-ddy)/(c_0*dt+ddy))*(ez[:,1]-ez_n[:,0])
           #2nd
           # ez[1:-1,0] = -ez_n_1[1:-1,1]-((ddx-(c_0*dt))/(ddx+(c_0*dt)))*(ez[1:-1,1]+ez_n_1[1:-1,0]) \
           #     + (2*ddx/(ddx+(c_0*dt)))*(ez_n[1:-1,0]+ez_n[1:-1,1]) \
           #     + ((ddx*(c_0*dt)**2)/(2*(ddy**2)*(ddx+c_0*dt)))*\
           #         (ez_n[2:,0]-2*ez_n[1:-1,0]+ez_n[:-2,0]+\
           #           ez_n[2:,1]-2*ez_n[1:-1,1]+ez_n[:-2,1])
           #Right
           #1st
           ez[Kx,:]=ez_n[Kx-1,:] +((c_0*dt-ddx)/(c_0*dt+ddx))*(ez[Kx-1,:]-ez_n[Kx,:])
           #2nd
           # ez[Kx,1:-1] = -ez_n_1[Kx-1,1:-1]-((ddx-(c_0*dt))/(ddx+(c_0*dt)))*(ez[Kx-1,1:-1]+ez_n_1[Kx,1:-1]) \
           #     + (2*ddx/(ddx+(c_0*dt)))*(ez_n[Kx,1:-1]+ez_n[Kx-1,1:-1]) \
           #     + ((ddx*(c_0*dt)**2)/(2*(ddy**2)*(ddx+c_0*dt)))*\
           #         (ez_n[Kx,2:]-2*ez_n[Kx,1:-1]+ez_n[Kx,:-2]+\
           #           ez_n[Kx-1,2:]-2*ez_n[Kx-1,1:-1]+ez_n[Kx-1,:-2])
           #Top
           #1st
           ez[:,Ky]=ez_n[:,Ky-1] +((c_0*dt-ddy)/(c_0*dt+ddy))*(ez[:,Ky-1]-ez_n[:,Ky])
           #2nd
           # ez[1:-1,Ky] = -ez_n_1[1:-1,Ky-1]-((ddx-(c_0*dt))/(ddx+(c_0*dt)))*(ez[1:-1,Ky-1]+ez_n_1[1:-1,Ky]) \
           #     + (2*ddx/(ddx+(c_0*dt)))*(ez_n[1:-1,Ky]+ez_n[1:-1,Ky-1]) \
           #     + ((ddx*(c_0*dt)**2)/(2*(ddy**2)*(ddx+c_0*dt)))*\
           #         (ez_n[2:,Ky]-2*ez_n[1:-1,Ky]+ez_n[:-2,Ky]+\
           #           ez_n[2:,Ky-1]-2*ez_n[1:-1,Ky-1]+ez_n[:-2,Ky-1])
           
           hy = hy + (dt/(mu_0*ddx))*(ez[1:,:]-ez[:-1,:])
           hx = hx - (dt/(mu_0*ddy))*(ez[:,1:]-ez[:,:-1])
       
       MEOW = np.int64(.7/c_0/dt)
       # Probe to collect input signal in time domain
       #if n<ddx/c_0/dt*kstart # Time index set so I don't overlap input and reflected signals 
       if n<np.int64(700):
           geninput[n] = round(ez[probe_input,int(Ky/2)].real,4) # Watch the index if you change the spatial grid
       else:
           geninput[n] = 0
       # Probes to collect reflected and transmitted signals in time domain
       if n>np.int64(700):
           trans[n] = round(ez[probe_transmittedx,probe_transmittedy].real,4)
           refl[n] = round(ez[probe_reflectedx, probe_reflectedy].real,4) # Watch the index if you change the spatial grid
       else:
           trans[n] = 0
           refl[n] = 0  
           
       ez_n_1 = ez_n.copy() # data for t = n-1
       ez_n = ez.copy()     # data for t = n
       if n <= 3000:
           if (n%5 == 0) and plotEz == True:
               real = appending2D(ez,Kx) #geninput+refl+trans
               animatedPlot.append(real)
           if (n%5 == 0) and plotH == True:
               if letter == "y":
                   HH[:-1,:] = hy[:,:]
               if letter == "x":
                   HH[:,:-1] = hx[:,:]
               animatedPlotH.append(HH)

if typeofinput[1] == 1:
    # ME-FDTD parameters
    aa=[]
    bb = []
    b= []
    D = []
    F = []
    B = []
    C = [] 
    a = []
    DD = []
    FF = []
    BB = []
    CC = []
    for i in range(0,len(wp)):
        for k in range(0,len(wp[i])):
            a.append((cmath.sqrt((nu[i,k]**2)-(4*(wp[i,k]**2))))/2) # This piece of eigenvalue can be complex ...
            b.append(nu[i,k]/2)
            DD.append((a[-1]+b[-1])/(2*a[-1])*cmath.exp((a[-1]-b[-1])*dt) + ((a[-1]-b[-1])/(2*a[-1])*cmath.exp((-a[-1]-b[-1])*dt)))
            FF.append((cmath.exp((a[-1]-b[-1])*dt) - cmath.exp((-a[-1]-b[-1])*dt))/(2*a[-1]))
            BB.append((a[-1]+b[-1])/(2*a[-1]*(a[-1]-b[-1]))*(cmath.exp((a[-1]-b[-1])*dt)-1) - (a[-1]-b[-1])/(2*a[-1]*(a[-1]+b[-1]))*(cmath.exp((-a[-1]-b[-1])*dt)-1))
            CC.append(1/(2*a[-1]*(a[-1]-b[-1]))*(cmath.exp((a[-1]-b[-1])*dt)-1) + 1/(2*a[-1]*(a[-1]+b[-1]))*(cmath.exp((-a[-1]-b[-1])*dt)-1))
            if i==150 and k == 20:
                HELPa = (cmath.sqrt((nu[i,k]**2)-(4*(wp[i,k]**2))))/2
                HELPb = nu[i,k]/2
                HELPFF = ((cmath.exp((a[-1]-b[-1])*dt) - cmath.exp((-a[-1]-b[-1])*dt))/(2*a[-1]))
        aa.append(a)
        bb.append(b)
        D.append(DD)
        F.append(FF)
        B.append(BB)
        C.append(CC)
        a = []
        b = []
        DD = []
        FF = []
        BB = []
        CC = []
    
    # # Calculate ME-FDTD parameters
    # a = np.power((np.power(nu,2,dtype = complex))-(4*np.power(wp,2,dtype = complex)),0.5)/2 # This piece of eigenvalue can be complex ...
    # b = (nu/2)
    # D = (a+b)/(2*a)*np.exp((a-b)*dt) + ((a-b)/(2*a)*np.exp((-a-b)*dt))
    # F = np.exp((a-b)*dt) - np.exp((-a-b)*dt)/(2*a)
    # B = (a+b)/(2*a*(a-b))*(np.exp((a-b)*dt)-1) - (a-b)/(2*a*(a+b))*(np.exp((-a-b)*dt)-1)
    # C = 1/(2*a*(a-b))*(np.exp((a-b)*dt)-1) + 1/(2*a*(a+b))*(np.exp((-a-b)*dt)-1)
    
    probe_input = 5
    probe_reflectedx = int(0.005/ddx)
    probe_transmittedx = int(0.075/ddx)   
    probe_reflectedy = int(Ky/2)
    probe_transmittedy = int(Ky/2)
    
    for n in range(0,nsteps): 
        if n%100 == 0:
            print("this is n %0.0f" %n)
        # #Before the plasma   
        # ez[1:kstart-1,1:kbottom-1] = ez[1:kstart-1,1:kbottom-1] +dt/eps_0 * \
        #                                             ((hy[1:kstart-1,1:kbottom-1]-hy[:int(kstart-2),1:kbottom-1])/ddx - \
        #                                               (hx[1:kstart-1,1:kbottom-1]-hx[1:kstart-1,:int(kbottom-2)])/ddy)
        ez[1:kstart,1:kbottom] = ez[1:kstart,1:kbottom] +dt/eps_0 * \
                                                    ((hy[1:kstart,1:kbottom]-hy[:kstart-1,1:kbottom])/ddx - \
                                                      (hx[1:kstart,1:kbottom]-hx[1:kstart,:kbottom-1])/ddy)
                                                         
        #Before the plasma Above the main square  
        ez[1:kstart,kbottom:-1] = ez[1:kstart,kbottom:-1] +dt/eps_0 * \
                                                    ((hy[1:kstart,kbottom:-1]-hy[:kstart-1,kbottom:-1])/ddx - \
                                                      (hx[1:kstart,kbottom:]-hx[1:kstart,kbottom-1:-1])/ddy)
                                                        
        #Before the plasma to the right of the main square 
        ez[kstart:-1,1:kbottom] = ez[kstart:-1,1:kbottom] +dt/eps_0 * \
                                                    ((hy[kstart:,1:kbottom]-hy[kstart-1:-1,1:kbottom])/ddx - \
                                                      (hx[kstart:-1,1:kbottom]-hx[kstart:-1,:kbottom-1])/ddy)
        
        # Source Pulse
        if n <= 750:
            pulse = -amplitude*(t0-Time)/spread*np.exp(-((t0-Time)/spread)**2) # Derivative Gaussian input signal
            #pulse = 40*np.sin(2*np.pi*3e9*Time) # Sine input with single frequency
            # pulse = 10
        else: 
            pulse = 0
        pulseArray[n] = pulse
        ez[4,:]=ez[4,:]+pulse
        Time+=dt
        
        for i in range(kstart,kend+1):
            for k in range(kbottom,ktop+1):
                # Through plasma with ME-FDTD formulation for Ez field   
                ezc = D[i][k]*ez[i,k] + B[i][k]/eps_0 * ((hy[i,k]-hy[i-1,k])/ddx - (hx[i,k]- hx[i,k-1])/ddy) - \
                                F[i][k]/eps_0*jj[i][k]
                     
                #1D
                # j[k] = (D[k] - nu[k]*F[k] + (wp[k]**2)*C[k]*F[k]/B[k])*j[k] + \
                #     (eps_0*(wp[k]**2)*C[k]/B[k])*ez[k] +\
                #     (eps_0*(wp[k]**2)*F[k] - eps_0*(wp[k]**2)*C[k]*D[k]/B[k])*ez_0[k]
                
                # j = (D.real - nu*F.real + (wp**2)*C.real*F.real/B.real)*j + (eps_0*(wp**2)*C.real/B.real)*ez + (eps_0*(wp**2)*F.real - eps_0*(wp**2)*C.real*D.real/B.real)*ez_0
        
                j[k] = (D[i][k] - nu[i,k]*F[i][k] + (wp[i,k]**2)*C[i][k]*F[i][k]/B[i][k])*jj[i][k] +\
                    (eps_0*(wp[i,k]**2)*C[i][k]/B[i][k])*ez[i,k] +\
                    (eps_0*(wp[i,k]**2)*F[i][k] - eps_0*(wp[i,k]**2)*C[i][k]*D[i][k]/B[i][k])*ez_0[i,k]
                if ezc.imag == 0j:
                    ezc = ezc.real
                ez[i,k] = ezc
                # if n >=400 and n<1000:
                #     print("This is j: {:.2f}".format(j[k]))
            jj[i] = j
            j = [0]*round(Ky+1)
        # # Through plasma with ME-FDTD formulation for Ez field   
        # ez[kstart:kend,kbottom:ktop] = \
        #     D[kstart:kend,kbottom:ktop].real*ez[kstart:kend,kbottom:ktop] + \
        #         B[kstart:kend,kbottom:ktop].real/eps_0 * \
        #             ((hy[kstart:kend,kbottom:ktop]-\
        #               hy[kstart-1:kend-1,kbottom:ktop])/ddx - \
        #               (hx[kstart:kend,kbottom:ktop]- \
        #               hx[kstart:kend,kbottom-1:ktop-1])/ddy) - \
        #                 F[kstart:kend,kbottom:ktop].real/eps_0*j[kstart:kend,kbottom:ktop]
        
        # # j = (D.real - nu*F.real + (wp**2)*C.real*F.real/B.real)*j + (eps_0*(wp**2)*C.real/B.real)*ez + (eps_0*(wp**2)*F.real - eps_0*(wp**2)*C.real*D.real/B.real)*ez_0

        # j[kstart:kend,kbottom:ktop] = (D[kstart:kend,kbottom:ktop].real - \
        #     nu[kstart:kend,kbottom:ktop]*F[kstart:kend,kbottom:ktop].real + \
        #         (wp[kstart:kend,kbottom:ktop]**2)*C[kstart:kend,kbottom:ktop].real* \
        #             F[kstart:kend,kbottom:ktop].real/B[kstart:kend,kbottom:ktop].real)*\
        #               j[kstart:kend,kbottom:ktop] + eps_0*(wp[kstart:kend,kbottom:ktop]**2)*\
        #                   C[kstart:kend,kbottom:ktop].real/B[kstart:kend,kbottom:ktop].real*\
        #                       ez[kstart:kend,kbottom:ktop] + (eps_0*(wp[kstart:kend,kbottom:ktop]**2)*\
        #                           F[kstart:kend,kbottom:ktop].real - eps_0*(wp[kstart:kend,kbottom:ktop]**2)*\
        #                               C[kstart:kend,kbottom:ktop].real*D[kstart:kend,kbottom:ktop].real/ \
        #                                   B[kstart:kend,kbottom:ktop].real)*ez_0[kstart:kend,kbottom:ktop]
        
        ez_0 = ez.copy() # Store current time step Ez field for next iteration   
    
        # ezRight = ez[kend,ktop]
        # #After the plasma top 
        ez[kstart:-1,ktop+1:-1] = ez[kstart:-1,ktop+1:-1] +dt/eps_0 * \
            ((hy[kstart:,ktop+1:-1]-\
              hy[kstart-1:-1,ktop+1:-1])/ddx - \
              (hx[kstart:-1,ktop+1:]-\
              hx[kstart:-1,ktop:-1])/ddy)
        
        # #Fix the right corner of the plasma
        # ez[kend,ktop] = ezRight 
        
        # #After the plasma  down
        ez[kend+1:-1,kbottom:ktop+1] = ez[kend+1:-1,kbottom:ktop+1] +dt/eps_0 * \
            ((hy[kend+1:,kbottom:ktop+1]-\
              hy[kend:-1,kbottom:ktop+1])/ddx - \
              (hx[kend+1:-1,kbottom:ktop+1]-\
              hx[kend+1:-1,kbottom-1:ktop])/ddy)
        
        #Mur Bondary Conditions from book 1st and second order
        #Left
        #1st
        ez[0,:] = ez_n[1,:] +((c_0*dt-ddx)/(c_0*dt+ddx))*(ez[1,:]-ez_n[0,:])
        #2nd
        # ez[0,1:-1] = -ez_n_1[1,1:-1]-((ddx-(c_0*dt))/(ddx+(c_0*dt)))*(ez[1,1:-1]+ez_n_1[0,1:-1]) \
        #     + (2*ddx/(ddx+(c_0*dt)))*(ez_n[0,1:-1]+ez_n[1,1:-1]) \
        #     + ((ddx*(c_0*dt)**2)/(2*(ddy**2)*(ddx+c_0*dt)))*\
        #         (ez_n[0,2:]-2*ez_n[0,1:-1]+ez_n[0,:-2]+\
        #           ez_n[1,2:]-2*ez_n[1,1:-1]+ez_n[1,:-2])
        #Bottom
        #1st
        ez[:,0] = ez_n[:,1] +((c_0*dt-ddy)/(c_0*dt+ddy))*(ez[:,1]-ez_n[:,0])
        #2nd
        # ez[1:-1,0] = -ez_n_1[1:-1,1]-((ddx-(c_0*dt))/(ddx+(c_0*dt)))*(ez[1:-1,1]+ez_n_1[1:-1,0]) \
        #     + (2*ddx/(ddx+(c_0*dt)))*(ez_n[1:-1,0]+ez_n[1:-1,1]) \
        #     + ((ddx*(c_0*dt)**2)/(2*(ddy**2)*(ddx+c_0*dt)))*\
        #         (ez_n[2:,0]-2*ez_n[1:-1,0]+ez_n[:-2,0]+\
        #           ez_n[2:,1]-2*ez_n[1:-1,1]+ez_n[:-2,1])
        #Right
        #1st
        ez[Kx,:]=ez_n[Kx-1,:] +((c_0*dt-ddx)/(c_0*dt+ddx))*(ez[Kx-1,:]-ez_n[Kx,:])
        #2nd
        # ez[Kx,1:-1] = -ez_n_1[Kx-1,1:-1]-((ddx-(c_0*dt))/(ddx+(c_0*dt)))*(ez[Kx-1,1:-1]+ez_n_1[Kx,1:-1]) \
        #     + (2*ddx/(ddx+(c_0*dt)))*(ez_n[Kx,1:-1]+ez_n[Kx-1,1:-1]) \
        #     + ((ddx*(c_0*dt)**2)/(2*(ddy**2)*(ddx+c_0*dt)))*\
        #         (ez_n[Kx,2:]-2*ez_n[Kx,1:-1]+ez_n[Kx,:-2]+\
        #           ez_n[Kx-1,2:]-2*ez_n[Kx-1,1:-1]+ez_n[Kx-1,:-2])
        #Top
        #1st
        ez[:,Ky]=ez_n[:,Ky-1] +((c_0*dt-ddy)/(c_0*dt+ddy))*(ez[:,Ky-1]-ez_n[:,Ky])
        #2nd
        # ez[1:-1,Ky] = -ez_n_1[1:-1,Ky-1]-((ddx-(c_0*dt))/(ddx+(c_0*dt)))*(ez[1:-1,Ky-1]+ez_n_1[1:-1,Ky]) \
        #     + (2*ddx/(ddx+(c_0*dt)))*(ez_n[1:-1,Ky]+ez_n[1:-1,Ky-1]) \
        #     + ((ddx*(c_0*dt)**2)/(2*(ddy**2)*(ddx+c_0*dt)))*\
        #         (ez_n[2:,Ky]-2*ez_n[1:-1,Ky]+ez_n[:-2,Ky]+\
        #           ez_n[2:,Ky-1]-2*ez_n[1:-1,Ky-1]+ez_n[:-2,Ky-1])
        
        hy = hy + (dt/(mu_0*ddx))*(ez[1:,:]-ez[:-1,:])
        hx = hx - (dt/(mu_0*ddy))*(ez[:,1:]-ez[:,:-1])
        
        MEOW = np.int64(.7/c_0/dt)
        # Probe to collect input signal in time domain
        #if n<ddx/c_0/dt*kstart # Time index set so I don't overlap input and reflected signals 
        if n<np.int64(750):
            geninput[n] = round(ez[probe_input,int(Ky/2)].real,4) # Watch the index if you change the spatial grid
        else:
            geninput[n] = 0
        # Probes to collect reflected and transmitted signals in time domain
        if n>np.int64(750):
            trans[n] = round(ez[int(probe_transmittedx),int(probe_transmittedy)].real,4)
            refl[n] = round(ez[int(probe_reflectedx), int(probe_reflectedy)].real,4) # Watch the index if you change the spatial grid
        else:
            trans[n] = 0
            refl[n] = 0  
            
        ez_n_1 = ez_n.copy() # data for t = n-1
        ez_n = ez.copy()     # data for t = n
        if n <= 3500:
            if (n%10 == 0) and plotEz == True:
                real = appending2D(ez,Kx) #geninput+refl+trans
                animatedPlot.append(real)
            if (n%10 == 0) and plotH == True:
                if letter == "y":
                    HH[:-1,:] = hy[:,:]
                if letter == "x":
                    HH[:,:-1] = hx[:,:]
                animatedPlotH.append(HH)

if typeofinput[2] == 1:
    for n in range(0,nsteps): 
        if n%100 == 0:
            print("this is n %0.0f" %n)
        
        #Recalculate Electric Field
        ez[1:-1,1:-1] = ez[1:-1,1:-1] +dt/eps_0 * ((hy[1:,1:-1]-hy[:-1,1:-1])/ddx - \
                                                     (hx[1:-1,1:]-hx[1:-1,:-1])/ddy)
        
        # Source Pulse
        if n <= 1000:
            pulse = -amplitude*(t0-Time)/spread*np.exp(-((t0-Time)/spread)**2) # Derivative Gaussian input signal
            # pulse = 40*np.sin(2*np.pi*3e9*Time) # Sine input with single frequency
            #pulse = 10
        else: 
            pulse = 0
        pulseArray[n] = pulse
        
        ez[10,:]=ez[10,:]+pulse
        Time+=dt
        
        #Mur Bondary Conditions from book 1st and second order
        #Left
        #1st
        ez[0,:] = ez_n[1,:] +((c_0*dt-ddx)/(c_0*dt+ddx))*(ez[1,:]-ez_n[0,:])
        #2nd
        # ez[0,1:-1] = -ez_n_1[1,1:-1]-((ddx-(c_0*dt))/(ddx+(c_0*dt)))*(ez[1,1:-1]+ez_n_1[0,1:-1]) \
        #     + (2*ddx/(ddx+(c_0*dt)))*(ez_n[0,1:-1]+ez_n[1,1:-1]) \
        #     + ((ddx*(c_0*dt)**2)/(2*(ddy**2)*(ddx+c_0*dt)))*\
        #         (ez_n[0,2:]-2*ez_n[0,1:-1]+ez_n[0,:-2]+\
        #           ez_n[1,2:]-2*ez_n[1,1:-1]+ez_n[1,:-2])
        #Bottom
        #1st
        ez[:,0] = ez_n[:,1] +((c_0*dt-ddy)/(c_0*dt+ddy))*(ez[:,1]-ez_n[:,0])
        #2nd
        # ez[1:-1,0] = -ez_n_1[1:-1,1]-((ddx-(c_0*dt))/(ddx+(c_0*dt)))*(ez[1:-1,1]+ez_n_1[1:-1,0]) \
        #     + (2*ddx/(ddx+(c_0*dt)))*(ez_n[1:-1,0]+ez_n[1:-1,1]) \
        #     + ((ddx*(c_0*dt)**2)/(2*(ddy**2)*(ddx+c_0*dt)))*\
        #         (ez_n[2:,0]-2*ez_n[1:-1,0]+ez_n[:-2,0]+\
        #           ez_n[2:,1]-2*ez_n[1:-1,1]+ez_n[:-2,1])
        #Right
        #1st
        ez[Kx,:]=ez_n[Kx-1,:] +((c_0*dt-ddx)/(c_0*dt+ddx))*(ez[Kx-1,:]-ez_n[Kx,:])
        #2nd
        # ez[Kx,1:-1] = -ez_n_1[Kx-1,1:-1]-((ddx-(c_0*dt))/(ddx+(c_0*dt)))*(ez[Kx-1,1:-1]+ez_n_1[Kx,1:-1]) \
        #     + (2*ddx/(ddx+(c_0*dt)))*(ez_n[Kx,1:-1]+ez_n[Kx-1,1:-1]) \
        #     + ((ddx*(c_0*dt)**2)/(2*(ddy**2)*(ddx+c_0*dt)))*\
        #         (ez_n[Kx,2:]-2*ez_n[Kx,1:-1]+ez_n[Kx,:-2]+\
        #           ez_n[Kx-1,2:]-2*ez_n[Kx-1,1:-1]+ez_n[Kx-1,:-2])
        #Top
        #1st
        ez[:,Ky]=ez_n[:,Ky-1] +((c_0*dt-ddy)/(c_0*dt+ddy))*(ez[:,Ky-1]-ez_n[:,Ky])
        #2nd
        # ez[1:-1,Ky] = -ez_n_1[1:-1,Ky-1]-((ddx-(c_0*dt))/(ddx+(c_0*dt)))*(ez[1:-1,Ky-1]+ez_n_1[1:-1,Ky]) \
        #     + (2*ddx/(ddx+(c_0*dt)))*(ez_n[1:-1,Ky]+ez_n[1:-1,Ky-1]) \
        #     + ((ddx*(c_0*dt)**2)/(2*(ddy**2)*(ddx+c_0*dt)))*\
        #         (ez_n[2:,Ky]-2*ez_n[1:-1,Ky]+ez_n[:-2,Ky]+\
        #           ez_n[2:,Ky-1]-2*ez_n[1:-1,Ky-1]+ez_n[:-2,Ky-1])
                   
        hy = hy + (dt/(mu_0*ddx))*(ez[1:,:]-ez[:-1,:])
        hx = hx - (dt/(mu_0*ddy))*(ez[:,1:]-ez[:,:-1])
        
        # Probe to collect input signal in time domain
        #if n<ddx/c_0/dt*kstart # Time index set so I don't overlap input and reflected signals 
        if n<np.int64(.3/c_0/dt):
            geninput[n] = round(ez[int(probe_input),int(probe_input)].real,4) # Watch the index if you change the spatial grid
        else:
            geninput[n] = 0
        # Probes to collect reflected and transmitted signals in time domain
        if n>np.int64(.3/c_0/dt):
            trans[n] = round(ez[int(probe_transmittedx),int(probe_transmittedy)].real,4)
            refl[n] = round(ez[int(probe_reflectedx), int(probe_reflectedy)].real,4) # Watch the index if you change the spatial grid
        else:
            trans[n] = 0
            refl[n] = 0  
            
        ez_n_1 = ez_n.copy() # data for t = n-1
        ez_n = ez.copy()     # data for t = n
        
        # Animation data collection (will not collect if the animation is off or =0)
        if n <= 3000:
            if (n%5 == 0) and plotEz == True:
                real = appending2D(ez,Kx) #geninput+refl+trans
                animatedPlot.append(real)
            if (n%5 == 0) and plotH == True:
                if letter == "y":
                    HH[:-1,:] = hy[:,:]
                if letter == "x":
                    HH[:,:-1] = hx[:,:]
                animatedPlotH.append(HH)

if typeofinput[3] == 1:
    a = np.sqrt((nu[kstart:kend,kbottom:ktop])**2-4*(wp[kstart:kend,kbottom:ktop])**2,dtype=complex)/2
    b = nu[kstart:kend,kbottom:ktop]/2
    D = (a+b)/(2*a) * np.exp((a-b)*dt) + (a-b)/(2*a) * np.exp((-a-b)*dt)
    F = (np.exp((a-b)*dt) - np.exp((-a-b)*dt))/(2*a)
    B = (a+b) / (2*a*(a-b)) * (np.exp((a-b) * dt)-1) - (a-b)/(2*a*(a+b))*(np.exp((-a-b)*dt)-1)
    C = 1/(2*a*(a-b))*(np.exp((a-b)*dt)-1) + 1/(2*a*(a+b))*(np.exp((-a-b)*dt)-1)
    j = np.zeros_like(F,dtype = complex)
    
    probe_input = 4
    probe_reflectedx = 3 #int(0.0375/ddx-4)
    probe_transmittedx = kend+4
    probe_reflectedy = int(Ky/2)
    probe_transmittedy = int(Ky/2)
    
    energyinput=np.zeros(nsteps)
    energyplasma=np.zeros(nsteps)
    energytrans=np.zeros(nsteps)
    energyrefl=np.zeros(nsteps)
    powerinput=np.zeros(nsteps)
    powerplasma=np.zeros(nsteps)
    powertrans=np.zeros(nsteps)
    powerrefl=np.zeros(nsteps)
    
    # start = kstart
    # end = kend
    # top = ktop
    # bottom = kbottom
    
    for n in range(0,nsteps): 
        if n%100 == 0:
            print("this is n %0.0f" %n)
        
        #Save the old
        ezOLD = ez.copy()
        # Recalculate Electric Field  
        ez[1:-1,1:-1] = ez[1:-1,1:-1] +dt/eps_0 * ((hy[1:,1:-1]-hy[:-1,1:-1])/ddx - \
                                                     (hx[1:-1,1:]-hx[1:-1,:-1])/ddy)
        
        # Source Pulse
        # if n <= 800:
        #     pulse = -amplitude*(t0-Time)/spread*np.exp(-((t0-Time)/spread)**2) # Derivative Gaussian input signal
        #     #pulse = 40*np.sin(2*np.pi*3e9*Time) # Sine input with single frequency
        #     # pulse = 10
        # else: 
        #     pulse = 0
        pulse = -amplitude*(t0-Time)/spread*np.exp(-((t0-Time)/spread)**2) # Derivative Gaussian input signal
        pulseArray[n] = pulse
        ez[4,:]=ez[4,:]+pulse
        Time+=dt
        
        # Recalculate Electric Field inside plasma
        # Through plasma with ME-FDTD formulation for Ez field   
        ez[kstart:kend,kbottom:ktop] = \
            D*ezOLD[kstart:kend,kbottom:ktop] + B/eps_0 * \
                    ((hy[kstart:kend,kbottom:ktop]-\
                      hy[kstart-1:kend-1,kbottom:ktop])/ddx - \
                      (hx[kstart:kend,kbottom:ktop]- \
                      hx[kstart:kend,kbottom-1:ktop-1])/ddy) - \
                        F/eps_0*j
        
        # Recalculate Current inside plasma (zero outside the plasma)
        # j = (D.real - nu*F.real + (wp**2)*C.real*F.real/B.real)*j + (eps_0*(wp**2)*C.real/B.real)*ez + (eps_0*(wp**2)*F.real - eps_0*(wp**2)*C.real*D.real/B.real)*ez_0
        j = (D - nu[kstart:kend,kbottom:ktop]*F + \
                (wp[kstart:kend,kbottom:ktop]**2)*C*F/B)*j + \
                    eps_0*(wp[kstart:kend,kbottom:ktop]**2)*\
                          C/B*ezOLD[kstart:kend,kbottom:ktop] + \
                              (eps_0*(wp[kstart:kend,kbottom:ktop]**2)*\
                                  F - eps_0*(wp[kstart:kend,kbottom:ktop]**2)*\
                                      C*D/B)*ez_0[kstart:kend,kbottom:ktop]
        
        
        # for i in range(kstart,kend+1):
        #     ez[i,kbottom:ktop] = \
        #         (D[i][kbottom:ktop])*ezOLD[i,kbottom:ktop] + \
        #             (B[i][kbottom:ktop])* \
        #                 ((hy[i,kbottom:ktop]-\
        #                   hy[i-1,kbottom:ktop])/ddx - \
        #                   (hx[i,kbottom:ktop]- \
        #                   hx[i,kbottom-1:ktop-1])/ddy)/eps_0 - \
        #                     F[i][kbottom:ktop]*j[i,kbottom:ktop]/eps_0
            
        #     j[i,kbottom:ktop] = (D[i][kbottom:ktop] - \
        #         nu[i,kbottom:ktop]*F[i][kbottom:ktop] + \
        #             (wp[i,kbottom:ktop]**2)*C[i][kbottom:ktop]* \
        #                 F[i][kbottom:ktop]/B[i][kbottom:ktop])*\
        #                   j[i,kbottom:ktop] + eps_0*(wp[i,kbottom:ktop]**2)*\
        #                       C[i][kbottom:ktop]/B[i][kbottom:ktop]*\
        #                           ezOLD[i,kbottom:ktop] + (eps_0*(wp[i,kbottom:ktop]**2)*\
        #                               F[i][kbottom:ktop] - eps_0*(wp[i,kbottom:ktop]**2)*\
        #                                   C[i][kbottom:ktop]*D[i][kbottom:ktop]/ \
        #                                       B[i][kbottom:ktop])*ez_0[i,kbottom:ktop]
        
        # Save the Electric Field for the next Time step
        ez_0 = ez.copy() # Store current time step Ez field for next iteration   
        
        #Mur Bondary Conditions from book 1st and second order
        #Left
        #1st
        ez[0,:] = ez_n[1,:] +((c_0*dt-ddx)/(c_0*dt+ddx))*(ez[1,:]-ez_n[0,:])
        #2nd
        # ez[0,1:-1] = -ez_n_1[1,1:-1]-((ddx-(c_0*dt))/(ddx+(c_0*dt)))*(ez[1,1:-1]+ez_n_1[0,1:-1]) \
        #     + (2*ddx/(ddx+(c_0*dt)))*(ez_n[0,1:-1]+ez_n[1,1:-1]) \
        #     + ((ddx*(c_0*dt)**2)/(2*(ddy**2)*(ddx+c_0*dt)))*\
        #         (ez_n[0,2:]-2*ez_n[0,1:-1]+ez_n[0,:-2]+\
        #           ez_n[1,2:]-2*ez_n[1,1:-1]+ez_n[1,:-2])
        #Bottom
        #1st
        ez[:,0] = ez_n[:,1] +((c_0*dt-ddy)/(c_0*dt+ddy))*(ez[:,1]-ez_n[:,0])
        #2nd
        # ez[1:-1,0] = -ez_n_1[1:-1,1]-((ddx-(c_0*dt))/(ddx+(c_0*dt)))*(ez[1:-1,1]+ez_n_1[1:-1,0]) \
        #     + (2*ddx/(ddx+(c_0*dt)))*(ez_n[1:-1,0]+ez_n[1:-1,1]) \
        #     + ((ddx*(c_0*dt)**2)/(2*(ddy**2)*(ddx+c_0*dt)))*\
        #         (ez_n[2:,0]-2*ez_n[1:-1,0]+ez_n[:-2,0]+\
        #           ez_n[2:,1]-2*ez_n[1:-1,1]+ez_n[:-2,1])
        #Right
        #1st
        ez[Kx,:]=ez_n[Kx-1,:] +((c_0*dt-ddx)/(c_0*dt+ddx))*(ez[Kx-1,:]-ez_n[Kx,:])
        #2nd
        # ez[Kx,1:-1] = -ez_n_1[Kx-1,1:-1]-((ddx-(c_0*dt))/(ddx+(c_0*dt)))*(ez[Kx-1,1:-1]+ez_n_1[Kx,1:-1]) \
        #     + (2*ddx/(ddx+(c_0*dt)))*(ez_n[Kx,1:-1]+ez_n[Kx-1,1:-1]) \
        #     + ((ddx*(c_0*dt)**2)/(2*(ddy**2)*(ddx+c_0*dt)))*\
        #         (ez_n[Kx,2:]-2*ez_n[Kx,1:-1]+ez_n[Kx,:-2]+\
        #           ez_n[Kx-1,2:]-2*ez_n[Kx-1,1:-1]+ez_n[Kx-1,:-2])
        #Top
        #1st
        ez[:,Ky]=ez_n[:,Ky-1] +((c_0*dt-ddy)/(c_0*dt+ddy))*(ez[:,Ky-1]-ez_n[:,Ky])
        #2nd
        # ez[1:-1,Ky] = -ez_n_1[1:-1,Ky-1]-((ddx-(c_0*dt))/(ddx+(c_0*dt)))*(ez[1:-1,Ky-1]+ez_n_1[1:-1,Ky]) \
        #     + (2*ddx/(ddx+(c_0*dt)))*(ez_n[1:-1,Ky]+ez_n[1:-1,Ky-1]) \
        #     + ((ddx*(c_0*dt)**2)/(2*(ddy**2)*(ddx+c_0*dt)))*\
        #         (ez_n[2:,Ky]-2*ez_n[1:-1,Ky]+ez_n[:-2,Ky]+\
        #           ez_n[2:,Ky-1]-2*ez_n[1:-1,Ky-1]+ez_n[:-2,Ky-1])
        
        # Recalculate Magnetic field in both x and y directions
        hy = hy + (dt/(mu_0*ddx))*(ez[1:,:]-ez[:-1,:])
        hx = hx - (dt/(mu_0*ddy))*(ez[:,1:]-ez[:,:-1])
        
        MEOW = np.int64(0.4/c_0/dt)
        
        # Probe to collect input signal in time domain
        #if n<ddx/c_0/dt*kstart # Time index set so I don't overlap input and reflected signals 
        if n<1200: # 1800
            geninput[n] = round(ez[probe_input,int(Ky/2)].real,4) # Watch the index if you change the spatial grid
            energyinput[n] = eps_0*(np.sum(ez[probe_input,:])**2) 
            powerinput[n] = (q*(np.sum(ez[probe_input,:]))/len(ez[probe_input]))/dt
        else:
            geninput[n] = 0
            energyinput[n] = 0
            powerinput[n] = 0
        # Probes to collect reflected and transmitted signals in time domain
        if n>1200:
            trans[n] = round(ez[probe_transmittedx,probe_transmittedy].real,4)
            refl[n] = round(ez[probe_reflectedx, probe_reflectedy].real,4) # Watch the index if you change the spatial grid
            
            energyplasma[n] =eps_0*(np.sum(ez[kstart+50,:])**2)
            powerplasma[n] = (q*(np.sum(ez[kstart+50,:]))/len(ez[kstart+50]))/dt
            
            energytrans[n] =eps_0*(np.sum(ez[probe_transmittedx,:])**2)
            powertrans[n] = (q*(np.sum(ez[probe_transmittedx,:]))/len(ez[probe_transmittedx]))/dt
            
            energyrefl[n] = eps_0*(np.sum(ez[probe_reflectedx,:])**2)
            powerrefl[n] = (q*(np.sum(ez[probe_reflectedx,:]))/len(ez[probe_reflectedx]))/dt
        else:
            trans[n] = 0
            refl[n] = 0 
            
            energyplasma[n] = 0
            powerplasma[n] = 0
            
            energytrans[n] = 0
            powertrans[n] = 0
            
            energyrefl[n] = 0
            powerrefl[n] = 0
          
        ez_n_1 = ez_n.copy() # data for t = n-1  (previous previous time step)
        ez_n = ez.copy()     # data for t = n    (Previous time step)
        
        # Animation data collection (will not collect if the animation is off or =0)
        if n <= 6000 and Want2Animate!=0:
            if (n%10 == 0) and plotEz == True:
                real = appending2D(ez,Kx) #geninput+refl+trans
                animatedPlot.append(real)
            if (n%10 == 0) and plotH == True:
                if letter == "y":
                    HH[:-1,:] = hy[:,:]
                if letter == "x":
                    HH[:,:-1] = hx[:,:]
                animatedPlotH.append(HH)
    dBmax1 = max(powerinput)
    dBmin1 = min(powerinput)
    dBmax2 = max(powerplasma)
    dBmin2 = min(powerplasma)
    dBmax3 = max(powertrans)
    dBmin3 = min(powertrans)
    dBmax4 = max(powerrefl)
    dBmin4 = min(powerrefl)
    SignalLossReflected = 10*math.log10(abs(dBmax4/dBmax1))
    SignalLossTransmitted = 10*math.log10(abs(dBmax3/dBmax1))
    SignalLossPlasma = 10*math.log10(abs(dBmax2/dBmax1))
    InitialSignal = (-1*(10*math.log10(abs(dBmax1))))+30

# Animation Functions
# Electric Field 
if (Want2Animate == 1) and plotEz == True:
    # Plot signal. Animation help from 1D Electromagnetic FDTD in Python by natsunoyuki from GitHub
    figure = plt.figure(figsize = [8,6]) 
    ax = figure.add_subplot(111,projection='3d')
    xx = np.linspace(0,Lx,Kx+1)
    yy = np.linspace(0,Ly,Ky+1)
    X, Y = np.meshgrid(xx,yy)
    # ax.contourf(X, Y, Z, zdir='z', offset=-100, cmap='coolwarm')
    # ax.contourf(X, Y, Z, zdir='x', offset=-40, cmap='coolwarm')
    # ax.contourf(X, Y, Z, zdir='y', offset=40, cmap='coolwarm')
    ax.set_xlabel('Propagation axis (m)')
    ax.set_ylabel('Propagation axis (m)')
    ax.set_zlabel('Signal (V/m)')
    def init():
        ax.set(xlim3d=[0, Lx], ylim3d=[0,Ly] ,  zlim3d=[-40, 40])
        #ax.view_init(0,90) #y z axis
        #ax.view_init(0,0) #x z axis
        #ax.plot(kstart*ddx,kbottom*ddy,[-50,50],color = 'm')
        #ax.plot(kend*ddx,ktop*ddy,[-50, 50],color = 'm')
        ax.grid()
        return line[0],
    #text((.06),47,'Free Space'); 
    #text((.22),47,'Plasma','Color','m'); 
    #text((.36),47,'Free Space');
    #animation
    # 
    # 
    # INPUTS: 
    #   n :
    def animations(frames, animatedPlot, line):
        if frames%100 == 0:
            print("this is the frame %0.0f" %frames)
        figure.suptitle('Time step = %0.0f' %(frames*10))
        #line.set_data(xx,yy)
        #line.set_3d_properties(zz[frames])
        line[0].remove()
        line[0] = ax.plot_surface(X,Y,animatedPlot[frames].T,cmap='viridis',label = "Input",vmin=-60,vmax=60)
        # ax.contourf(X, Y, animatedPlot[frames], zdir='z', offset=-100, cmap='coolwarm')
        # ax.contourf(X, Y, animatedPlot[frames], zdir='x', offset=-40, cmap='coolwarm')
        # ax.contourf(X, Y, animatedPlot[frames], zdir='y', offset=40, cmap='coolwarm')
        return line[0],
    line = [ax.plot_surface(X,Y,animatedPlot[0].T,cmap='viridis',label = "Input",vmin=-60,vmax=60)]    
    # ax.contourf(X, Y, animamatedPlot[0], zdir='x', offset=-40, cmap='coolwarm')
    # ax.contourf(X, Y, tedPlot[0], zdir='z', offset=-100, cmap='coolwarm')
    # ax.contourf(X, Y, anianimatedPlot[0], zdir='y', offset=40, cmap='coolwarm')
    # creating colorbar
    figure.colorbar(line[0], shrink=0.5, aspect=10)
    anim = animation.FuncAnimation(fig = figure, func = animations, interval = 400, frames = range(0,len(animatedPlot),2), fargs=(animatedPlot, line), init_func=init, blit = True) #len(ez) - 1
    anim.save("Plasma_2Dfdtd_animation.gif", writer = "pillow")
    plt.show()
elif (Want2Animate == 2) and plotEz == True:
    start = kstart*np.ones(Kx+1)
    end = kend*np.ones(Kx+1)
    # Plot signal. Animation help from 1D Electromagnetic FdTD in Python by natsunoyuki from GitHub
    figure = plt.figure() 
    ax = figure.add_subplot()
    xx = np.linspace(0,Lx,Kx+1)
    yy = np.linspace(0,Ly,Ky+1)
    X, Y = np.meshgrid(xx,yy)
    ax.set_xlabel('Propagation axis (m)')
    ax.set_ylabel('Propagation axis (m)')
    #Plasma line plotting
    plasmaline = np.outer([0]*(Kx+1),np.ones(Ky+1)) 
    plasmaline[int(kstart-2):int(kstart+2),kbottom:ktop]=-100000
    plasmaline[int(kend-2):int(kend+2),kbottom:ktop]=-1000000
    def init():
        ax.set(xlim=[0, Lx], ylim=[0, Ly]) 
        ax.grid()
        return line[0],
    
    #animation
    # 
    # 
    # INPUTS: 
    #   frames : the range set in the FuncAnimation function below
    #   animatedPlot: where the changing signal is saved to
    #   line: the variable that holds the pcolormesh info
    def animations(frames, animatedPlot, line):
        if frames%100 == 0:
            print("this is the frame %0.0f" %frames)
        figure.suptitle('Time step = %0.0f' %(frames*10))
        
        line[0] = ax.pcolormesh(X,Y,(animatedPlot[frames]+plasmaline).T,cmap='seismic',label = "Input",vmin=-80,vmax=80)
        return line[0],
    line = [ax.pcolormesh(X,Y,animatedPlot[0].T,cmap='seismic',label = "Input",vmin=-80,vmax=80)] 
    figure.colorbar(line[0], shrink=0.5, aspect=10)
    anim = animation.FuncAnimation(fig = figure, func = animations, interval = 300, frames = range(0,len(animatedPlot),10), fargs=(animatedPlot, line), init_func=init, blit = True) #len(ez) - 1
    anim.save("Plasma_2Dfdtd_animation.gif", writer = "pillow")
    plt.show()
else:
    animatedPlot = []
# Magnetic Field
if (Want2Animate == 1) and plotH == True:
   # Plot signal. Animation help from 1D Electromagnetic FDTD in Python by natsunoyuki from GitHub
   figure = plt.figure() #figsize = [8,10]
   ax = figure.add_subplot(111,projection='3d')
   xx = np.linspace(0,Lx,Kx+1)
   yy = np.linspace(0,Ly,Ky+1)
   X, Y = np.meshgrid(xx,yy)
   # ax.contourf(X, Y, Z, zdir='z', offset=-100, cmap='coolwarm')
   # ax.contourf(X, Y, Z, zdir='x', offset=-40, cmap='coolwarm')
   # ax.contourf(X, Y, Z, zdir='y', offset=40, cmap='coolwarm')
   ax.set_xlabel('Propagation axis (m)')
   ax.set_ylabel('Propagation axis (m)')
   ax.set_zlabel('Signal (V/m)')
   def init():
       ax.set(xlim3d=[0, Lx], ylim3d=[0,Ly] ,  zlim3d=[-40, 40])
       #ax.view_init(0,90) #y z axis
       #ax.view_init(0,0) #x z axis
       #ax.plot(kstart*ddx,kbottom*ddy,[-50,50],color = 'm')
       #ax.plot(kend*ddx,ktop*ddy,[-50, 50],color = 'm')
       ax.grid()
       return line[0],
   #text((.06),47,'Free Space'); 
   #text((.22),47,'Plasma','Color','m'); 
   #text((.36),47,'Free Space');
   #animation
   # 
   # 
   # INPUTS: 
   #   n :
   def animations(frames, animatedPlotH, line):
       if frames%100 == 0:
           print("this is the frame %0.0f" %frames)
       figure.suptitle('Time step = %0.0f' %(frames*5))
       #line.set_data(xx,yy)
       #line.set_3d_properties(zz[frames])
       line[0].remove()
       line[0] = ax.plot_surface(X,Y,animatedPlotH[frames].T,cmap='viridis',label = "Input")
       # ax.contourf(X, Y, animatedPlot[frames], zdir='z', offset=-100, cmap='coolwarm')
       # ax.contourf(X, Y, animatedPlot[frames], zdir='x', offset=-40, cmap='coolwarm')
       # ax.contourf(X, Y, animatedPlot[frames], zdir='y', offset=40, cmap='coolwarm')
       return line[0],
   line = [ax.plot_surface(X,Y,animatedPlotH[0].T,cmap='viridis',label = "Input")]    
   # ax.contourf(X, Y, animamatedPlot[0], zdir='x', offset=-40, cmap='coolwarm')
   # ax.contourf(X, Y, tedPlot[0], zdir='z', offset=-100, cmap='coolwarm')
   # ax.contourf(X, Y, anianimatedPlot[0], zdir='y', offset=40, cmap='coolwarm')
   figure.colorbar(line[0], shrink=0.5, aspect=10)
   anim = animation.FuncAnimation(fig = figure, func = animations, interval = 400, frames = range(0,len(animatedPlotH),2), fargs=(animatedPlotH, line), init_func=init, blit = True) #len(ez) - 1
   anim.save("Plasma_2Dfdtd_animation_Magnetic.gif", writer = "pillow")
   plt.show()
elif (Want2Animate == 2) and plotH == True:
    # Plot signal. Animation help from 1D Electromagnetic FdTD in Python by natsunoyuki from GitHub
    figure = plt.figure() #figsize = [8,10]
    ax = figure.add_subplot(111)
    # X = np.outer(np.linspace(0,Lx,Kx),np.ones(Kx+1))
    # X = X.copy().T
    # Y = np.outer(np.linspace(0,Ly,Ky+1),np.ones(Ky))
    xx = np.linspace(0,Lx,Kx+1)
    yy = np.linspace(0,Ly,Ky+1)
    X, Y = np.meshgrid(xx,yy)
    # ax.contourf(X, Y, Z, zdir='z', offset=-100, cmap='coolwarm')
    # ax.contourf(X, Y, Z, zdir='x', offset=-40, cmap='coolwarm')
    # ax.contourf(X, Y, Z, zdir='y', offset=40, cmap='coolwarm')
    def init():
        ax.set(xlim=[0, Lx], ylim=[0,Ly]) 
        #ax.plot(kstart*ddx,kbottom*ddy,[-50,50],color = 'm')
        #ax.plot(kend*ddx,ktop*ddy,[-50, 50],color = 'm')
        ax.grid()
        return line[0],
    #text((.06),47,'Free Space'); 
    #text((.22),47,'Plasma','Color','m'); 
    #text((.36),47,'Free Space');
    #animation
    # 
    # 
    # INPUTS: 
    #   n :
    def animations(frames, animatedPlotH, line):
        if frames%100 == 0:
            print("this is the frame %0.0f" %frames)
        figure.suptitle('Time step = %0.0f' %(frames*5))
        #line.set_data(xx,yy)
        #line.set_3d_properties(zz[frames])
        #line[0].remove()
        line[0] = ax.pcolormesh(X,Y,animatedPlotH[frames].T,cmap='viridis',label = "Input",vmin=-0.1,vmax=0.1)
        # ax.contourf(X, Y, animatedPlot[frames], zdir='z', offset=-100, cmap='coolwarm')
        # ax.contourf(X, Y, animatedPlot[frames], zdir='x', offset=-40, cmap='coolwarm')
        # ax.contourf(X, Y, animatedPlot[frames], zdir='y', offset=40, cmap='coolwarm')
        return line[0],
    line = [ax.pcolormesh(X,Y,animatedPlotH[0].T,cmap='viridis',label = "Input",vmin=-0.1,vmax=0.1)] 
    # ax.xlabel('Propagation axis (m)')
    # ax.zlabel('Propagation axis (m)')
    # ax.ylabel('Signal (V/m)')
    # ax.contourf(X, Y, animamatedPlot[0], zdir='x', offset=-40, cmap='coolwarm')
    # ax.contourf(X, Y, tedPlot[0], zdir='z', offset=-100, cmap='coolwarm')
    # ax.contourf(X, Y, anianimatedPlot[0], zdir='y', offset=40, cmap='coolwarm')
    figure.colorbar(line[0], shrink=0.5, aspect=10)
    anim = animation.FuncAnimation(fig = figure, func = animations, interval = 400, frames = range(0,len(animatedPlotH),5), fargs=(animatedPlotH, line), init_func=init, blit = True) #len(ez) - 1
    anim.save("Plasma_2Dfdtd_animation_Magnetic.gif", writer = "pillow")
    plt.show()
else:
    animatedPlotH = []
     
# This is the 3 graphs of the Input, Reflection and Transmission over the total number of time steps
if Figures2Print[4] == 1:
    xplotting = np.linspace(0,nsteps-1,nsteps)
    figure2 = plt.figure(figsize = [15,15])
    p1 = figure2.add_subplot(3,1,1)
    p2 = figure2.add_subplot(3,1,2)
    p3 = figure2.add_subplot(3,1,3)
    p1.plot(xplotting,geninput,'ok',linewidth=1,label = "Input")
    p2.plot(xplotting,refl,'ok',linewidth=1,label = "Reflected")
    p3.plot(xplotting,trans,'ok',linewidth=1,label = "Transmitted")
    p1.set_xlabel("TimeStep")
    p1.set_ylabel("Input V/m")
    p2.set_xlabel("TimeStep")
    p2.set_ylabel("Reflected V/m")
    p3.set_xlabel("TimeStep")
    p3.set_ylabel("Transmitted V/m")
    p1.set_xlim(0,nsteps)
    p2.set_xlim(0,nsteps)
    p3.set_xlim(0,nsteps)
    plt.show()

#This is the debugging version of the Input and Pulse and their zoomed graphs
if Figures2Print[5] == 1:
    xplotting = np.linspace(0,nsteps-1,nsteps)
    figure2 = plt.figure(figsize = [15,15])
    p1 = figure2.add_subplot(4,1,1)
    p2 = figure2.add_subplot(4,1,2)
    p3 = figure2.add_subplot(4,1,3)
    p4 = figure2.add_subplot(4,1,4)
    p1.plot(xplotting,geninput,'ok',linewidth=1,label = "Input Zoomed")
    p2.plot(xplotting,geninput,'ok',linewidth=1,label = "Input")
    p3.plot(xplotting,pulseArray,'ok',linewidth=1,label ="Input")
    p4.plot(xplotting,pulseArray,'ok',linewidth=1,label = "Pulse Zoomed")
    p1.set_xlabel("TimeStep")
    p1.set_ylabel("Input Zoomed")
    p2.set_xlabel("TimeStep")
    p2.set_ylabel("Input")
    p3.set_xlabel("TimeStep")
    p3.set_ylabel("Pulse")
    p4.set_xlabel("TimeStep")
    p4.set_ylabel("Pulse Zoomed")
    p1.set_xlim(0,1700)
    p4.set_xlim(0,1700)
    plt.show()

# Sampling frequency
Fs=1/dt
# Fourier transform of signals to the frequency domain
Y1 = fft(geninput)
Y2 = fft(trans)
Y3 = fft(refl)
f = Fs*np.linspace(0,1,len(Y1)) # frequency variable
# Plot spectrum of input signal
Py1 = Y1*np.conj(Y1)

#This is the Derivative Gaussian Pulse Distribution Graph
if Figures2Print[6] ==1:
    figure2, (plt1) = plt.subplots(1,1)
    plt1.plot(f/(10**9),abs(Py1))
    plt1.set_xlim(0, 11) 
    plt1.set_ylim(0, 1.2*max(abs(Py1)))
    plt1.set_xlabel('Frequency (GHz)')
    plt1.set_ylabel('PSD of Input Signal')
    figure2.suptitle('Derivative Gaussian Pulse Distribution')
    plt1.vlines([0.39, 1.550, 5.2], 0, 1.2*max(abs(Py1)), color = 'm', label = ['L-Band','S-Band','X-Band'])
    plt.figtext(0.19,0.85,'L-Band',horizontalalignment = 'center', verticalalignment = 'center', wrap = True, fontsize = 8, color = 'm')
    plt.figtext(0.27,0.85,'S-Band',horizontalalignment = 'center', verticalalignment = 'center', wrap = True, fontsize = 8, color = 'm')
    plt.figtext(0.526,0.85,'X-Band',horizontalalignment = 'center', verticalalignment = 'center', wrap = True, fontsize = 8, color = 'm')

    plt1.vlines([3.9, 6.2], 0, 1.2*max(abs(Py1)),color = 'b', label = ['C-Band','C-Band'])
    plt.figtext(0.44,0.85,'C-Band',horizontalalignment = 'center', verticalalignment = 'center', wrap = True, fontsize = 8, color = 'b')
    plt.figtext(0.6,0.85,'C-Band',horizontalalignment = 'center', verticalalignment = 'center', wrap = True, fontsize = 8, color = 'b')

    plt1.grid()
    plt.show()

# Calculate Transmission and Reflection coefficients (loss in dB)
magT1 = []
magR = []
T1 = Y2/Y1
R = Y3/Y1
for  v in range(0,len(T1)): #t
    #print("this is v %0.0f" %v)
    magT1.append(10*cmath.log10(abs(T1[v]*T1[v])))
    magR.append(10*cmath.log10(abs(R[v]*R[v])))
##
d = (kend-kstart)*ddx

# Plot reflected and transmission coefficients in the x
if Figures2Print[7] == 1:
    figure3, (ax1, ax2) = plt.subplots(2,1)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(1.2*min(np.real(magR)), 1)
    ax1.grid()
    ax1.set_xlabel("Frequency (GHz)")
    ax1.set_ylabel("Reflection coefficient\n magnitude (dB)")
    ax1.vlines([0.39, 1.550, 5.2],1.2*min(np.real(magR)), 1,color = 'm', label = ['L-Band','S-Band','X-Band'])
    ax1.vlines([3.9, 6.2],1.2*min(np.real(magR)), 1,color = 'b', label = ['C-Band','C-Band'])
    ax1.plot(f/(10**9),magR,'o'or ' ', c = 'k' ,ms = 3) 
    
    #Dividing the wHz exact by 10^9 to see better on the graph
    wHz_exactx_correct = [0]*len(wHz_exactx)
    for i in range(0,len(wHz_exactx)):
        wHz_exactx_correct[i] = wHz_exactx[i]/(10**9)
        
    ax1.plot(wHz_exactx_correct,magR_exactx,c = 'r') #'#8AF4FF'
    ax2.set_ylim(1.2*min(np.real(magT1)), 1)
    ax2.set_xlim(0, 10)
    ax2.grid()
    ax2.set_xlabel("Frequency (GHz)")
    ax2.set_ylabel('Transmission coefficient\n magnitude (dB)')
    ax2.vlines([0.39, 1.550, 5.2],1.2*min(np.real(magT1)), 1,color = 'm', label = ['L-Band','S-Band','X-Band'])
    ax2.vlines([3.9, 6.2],1.2*min(np.real(magT1)), 1,color = 'b', label = ['C-Band','C-Band'])
    ax2.plot(f/(10**9),magT1,'o'or ' ', c = 'k' ,ms = 3)
    ax2.plot(wHz_exactx_correct,magT_exactx,c = 'r')#'#8AF4FF'
    figure3.suptitle('Reflected and Transmission Coefficients (X)')
    plt.show()    

# Plot reflected and transmission coefficients in the Y
if Figures2Print[7] == 1:
    figure4, (ax1, ax2) = plt.subplots(2,1)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(1.2*min(np.real(magR)), 1)
    ax1.grid()
    ax1.set_xlabel("Frequency (GHz)")
    ax1.set_ylabel("Reflection coefficient\n magnitude (dB)")
    ax1.vlines([0.39, 1.550, 5.2],1.2*min(np.real(magR)), 1,color = 'm', label = ['L-Band','S-Band','X-Band'])
    ax1.vlines([3.9, 6.2],1.2*min(np.real(magR)), 1,color = 'b', label = ['C-Band','C-Band'])
    ax1.plot(f/(10**9),magR,'o'or ' ', c = 'k' ,ms = 3) 
    
    #Dividing the wHz exact by 10^9 to see better on the graph
    wHz_exacty_correct = [0]*len(wHz_exacty)
    for i in range(0,len(wHz_exacty)):
        wHz_exacty_correct[i] = wHz_exacty[i]/(10**9)
        
    ax1.plot(wHz_exacty_correct,magR_exacty,c = 'r') #'#8AF4FF'
    ax2.set_ylim(1.2*min(np.real(magT1)), 1)
    ax2.set_xlim(0, 10)
    ax2.grid()
    ax2.set_xlabel("Frequency (GHz)")
    ax2.set_ylabel('Transmission coefficient\n magnitude (dB)')
    ax2.vlines([0.39, 1.550, 5.2],1.2*min(np.real(magT1)), 1,color = 'm', label = ['L-Band','S-Band','X-Band'])
    ax2.vlines([3.9, 6.2],1.2*min(np.real(magT1)), 1,color = 'b', label = ['C-Band','C-Band'])
    ax2.plot(f/(10**9),magT1,'o'or ' ', c = 'k' ,ms = 3)
    ax2.plot(wHz_exacty_correct,magT_exacty,c = 'r')#'#8AF4FF'
    figure4.suptitle('Reflected and Transmission Coefficients (Y)')
    plt.show()    

# Plot power/energy coefficients
 
# if Figures2Print[8] ==1:
#     figure2, (plt1) = plt.subplots(1,1)
#     plt1.plot((powerdensityinput1))
#     plt1.set_xlim(0, 10000) 
#    # plt1.set_ylim( -1000, 1.2*max((powerdensityinput1)))
#     plt1.set_xlabel('Time(s)')
#     plt1.set_ylabel('Energy Density(J/cm^3')
#     figure2.suptitle('Energy Density in Region I')
#     plt1.grid()
#     plt.show()
# if Figures2Print[8] ==1:
#     figure2, (plt1) = plt.subplots(1,1)
#     plt1.plot((powerdensityinput2))
#     plt1.set_xlim(0, 10000) 
#     #plt1.set_ylim( -1000, 1.2*max((powerdensityinput2)))
#     plt1.set_xlabel('Time(s)')
#     plt1.set_ylabel('Energy Density(J/cm^3')
#     figure2.suptitle('Energy Density in Region II')
#     plt1.grid()
#     plt.show()
# if Figures2Print[8] ==1:
#     figure2, (plt1) = plt.subplots(1,1)
#     plt1.plot((powerdensityinput3))
#     plt1.set_xlim(0, 10000 ) 
#    #plt1.set_ylim(-1000 , 1.2*max((powerdensityinput3)))
#     plt1.set_xlabel('Time(s)')
#     plt1.set_ylabel('Energy Density(J/cm^3')
#     figure2.suptitle('Energy Density in Region III')
#     plt1.grid()
#     plt.show()

if Figures2Print[8] == 1:
    xplotting = np.linspace(0,nsteps-1,nsteps)
    figure2 = plt.figure(figsize = [15,14])
    figure2.suptitle('Energy Density', fontsize = 45)
    p1 = figure2.add_subplot(4,1,1)
    p2 = figure2.add_subplot(4,1,2)
    p3 = figure2.add_subplot(4,1,3)
    p4 = figure2.add_subplot(4,1,4)
    p1.plot(xplotting,energyinput,'ok',linewidth=1,label = "Input")
    p2.plot(xplotting,energyplasma,'ok',linewidth=1,label = "Inside Plasma")
    p3.plot(xplotting,energytrans,'ok',linewidth=1,label = "Transmitted")
    p4.plot(xplotting,energyrefl,'ok',linewidth=1,label = "Reflected")
    p1.set_xlabel("TimeStep")
    p1.set_ylabel("Energy Density Reg I")
    p2.set_xlabel("TimeStep")
    p2.set_ylabel("Energy Density Reg II")
    p3.set_xlabel("TimeStep")
    p3.set_ylabel("Energy Density Reg III")
    p4.set_xlabel("TimeStep")
    p4.set_ylabel("Energy Density Reg I (Reflected)")
    p1.set_xlim(0,nsteps/2)
    p2.set_xlim(0,nsteps/2)
    p3.set_xlim(0,nsteps/2)
    p4.set_xlim(0,nsteps/2)
    plt.show()

if Figures2Print[9] == 1:
    xplotting = np.linspace(0,int(nsteps)-1,int(nsteps))
    figure2 = plt.figure(figsize = [15,14])
    figure2.suptitle('Power', fontsize = 45)
    p1 = figure2.add_subplot(4,1,1)
    p2 = figure2.add_subplot(4,1,2)
    p3 = figure2.add_subplot(4,1,3)
    p4 = figure2.add_subplot(4,1,4)
    p1.plot(xplotting,powerinput,'ok',linewidth=1,label = "Power Input")
    p2.plot(xplotting,powerplasma,'ok',linewidth=1,label = "Power Inside Plasma")
    p3.plot(xplotting,powertrans,'ok',linewidth=1,label = "Power Transmitted")
    p4.plot(xplotting,powerrefl,'ok',linewidth=1,label = "Power Reflected")
    p1.set_xlabel("TimeStep")
    p1.set_ylabel("Power Reg I (Before Plasma)")
    p2.set_xlabel("TimeStep")
    p2.set_ylabel("Power Reg II (During Plasma)")
    p3.set_xlabel("TimeStep")
    p3.set_ylabel("Power Reg III (After Plasma)")
    p4.set_xlabel("TimeStep")
    p4.set_ylabel("Power Reg I (Reflected)")
    p1.set_xlim(0,nsteps*3/4)
    p2.set_xlim(0,nsteps*3/4)
    p3.set_xlim(0,nsteps*3/4)
    p4.set_xlim(0,nsteps*3/4)
    plt.figtext(0.7,0.85,'Initial Signal Power %0.2f dBm' %InitialSignal,horizontalalignment = 'center', verticalalignment = 'center', wrap = True, fontsize = 14, color = 'green')
    plt.figtext(0.7,0.65,'Signal Loss %0.2f dB' %SignalLossPlasma,horizontalalignment = 'center', verticalalignment = 'center', wrap = True, fontsize = 14, color = 'green')
    plt.figtext(0.7,0.45,'Signal Loss %0.2f dB' %SignalLossTransmitted,horizontalalignment = 'center', verticalalignment = 'center', wrap = True, fontsize = 14, color = 'green')
    plt.figtext(0.7,0.25,'Signal Loss %0.2f dB' %SignalLossReflected,horizontalalignment = 'center', verticalalignment = 'center', wrap = True, fontsize = 14, color = 'green')
    #plt.figtext(0.7,0.85,'Signal Loss',horizontalalignment = 'center', verticalalignment = 'center', wrap = True, fontsize = 14, color = 'green')
    plt.show()
    
    
    
    
    
    
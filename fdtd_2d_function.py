import numpy as np
from animate_fdtd import animateFDTD
import numba
from time import time
from tqdm import trange

"""Physical Constants"""
c_0 = 2.9975e+8 # speed of light in m/s
eps_0 = 8.854e-12 # permittivity of free space in F/m
mu_0 = 1.257e-6 # permeability of free space in H/m

USE_NUMBA = False


@numba.vectorize(["float64(float64,float64,float64,float64,float64,float64,float64)"],
    nopython=True,target='cpu')
def ez_op(ez,hy1,hy2,hx1,hx2,hyconst,hxconst):
    return ez + hyconst * (hy1-hy2) - hxconst * (hx1-hx2)

@numba.vectorize(["float64(float64,float64,float64,float64)"],
    nopython=True,target='cpu')
def h_op(h,ez1,ez2,const):
    return h + const * (ez1-ez2)

@numba.vectorize(["float64(float64,float64,float64,float64,float64,float64,float64,float64,float64,float64,float64,float64)"],
    nopython=True,target='cpu')
def ez_plasma_op(ez_0,hy1,hy2,hx1,hx2,D,B,F,j,hyconst,hxconst,Fconst):
    return D*ez_0 + B * (hyconst*(hy1-hy2) - hxconst*(hx1-hx2)) - F*Fconst*j

@numba.vectorize(["float64(float64,float64,float64,float64,float64,float64,float64,float64,float64)"],
    nopython=True,target='cpu')
def j_plasma_op(j,D,F,nu,wp,C,B,ez,ez_0):
    return j*(D-nu*F+wp**2*C*F/B) + (eps_0*wp**2*C/B)*ez + (eps_0*wp**2*F - eps_0*wp**2*C*D/B)*ez_0

def fdtd(nx,ny,dx,pxstart,pystart,wp,nu,input,nsteps,
    input_slices,poll_slices,
    dy=None,CFL=0.5,last_input_timestep=1000):
    """
    Function to propagate EM-Waves in 2D using FDTD with Matrix-Exponential Plasma Method
    Required Arguments:
    nx: number of cells in the x-direction
    ny: number of cells in the y-direction
    dx: cell size in x-direction (meters)
    pxstart: integer denoting x index where plasma starts. Requires pxstart+np.size(wp,0)<nx
    pystart: integer denoting y index where plasma starts. Requires pystart+np.size(wp,1)<ny
    wp: np 2D array of plasma frequencies (rad/s)
    nu: np 2D array of collision frequencies (hZ)
    input: function that takes as argument the elapsed time t and timestep dt in seconds, returns EM field strength (V/m)
    nsteps: number of FDTD timesteps to run
    input_slices: array of slice tuples denoting where the input should be added
    poll_slices: array of slice tuples denoting where to poll and return the input
    Optional Arguments:
    dy: cell size in y-direction (meters) Set equal to dx if not provided
    CFL: Courant-Friedrichs-Lewis Factor, defaults to 0.5
    last_input_timestep: stop input after this timestep
    """

    """Check that plasma dimensions are not too large, wp and nu are same dimension"""
    print(f'wp size 1 & 2: {np.size(wp,0)} {np.size(wp,1)}')
    assert(pxstart+np.size(wp,0) < nx)
    assert(pystart+np.size(wp,1) < ny)
    assert(wp.shape == nu.shape)
    pxend = pxstart+np.size(wp,0)-1 
    pyend = pystart+np.size(wp,1)-1

    """Set up Program Constants"""
    dy = dx if dy is None else dy
    ez = np.zeros((nx+1,ny+1))
    hx = np.zeros((nx+1,ny))
    hy = np.zeros((nx,ny+1))
    dt = CFL * min((dy,dx)) / c_0

    """Test - set up relative permittivity & permeability in a square in the center"""
    eps_rel = np.ones((nx+1, ny+1))
    mu_rel = np.ones((nx+1, ny+1))
    square_size=100
    sx,ex = int(nx/2)-int(square_size/2), int(nx/2)+int(square_size/2)
    sy,ey = int(ny/2)-int(square_size/2), int(ny/2)+int(square_size/2)
    eps_rel[sx:ex,sy:ey] *= 3000
    mu_rel[sx:ex,sy:ey] *= 1


    """Calculate Matrix-Exponential FDTD Plasma Parameters"""
    a = np.sqrt(nu**2-4*wp**2,dtype=complex)/2.
    b = nu/2.
    D = (a+b)/(2*a) * np.exp((a-b)*dt) + (a-b)/(2*a) * np.exp((-a-b)*dt)
    F = (np.exp((a-b)*dt) - np.exp((-a-b)*dt))/(2*a)
    B = (a+b) / (2*a*(a-b)) * (np.exp((a-b) * dt)-1) - (a-b)/(2*a*(a+b))*(np.exp((-a-b)*dt)-1)
    C = 1/(2*a*(a-b))*(np.exp((a-b)*dt)-1) + 1/(2*a*(a+b))*(np.exp((-a-b)*dt)-1)
    j = np.zeros_like(F,dtype = complex)


    ez_0 = np.copy(ez)

    """Set up polling data"""
    polled_data = [None] * len(poll_slices)
    for i, (row_slice, column_slice) in enumerate(poll_slices):
        polled_data[i] = np.zeros((nsteps,ez[row_slice,column_slice].shape[0],ez[row_slice,column_slice].shape[1]))


    """Main FDTD loop"""    
    start = time()
    for n in trange(nsteps):
        t = dt*n

        if(n%25==0):
            animateFDTD(n,nx,ny,dx,dy,ez,square_size)
            pass

        if(USE_NUMBA):
            """E_z outsize plasma field"""
            ez_op(ez[1:pxstart,pystart:pyend+1],
                hy[1:pxstart,pystart:pyend+1],
                hy[:pxstart-1,pystart:pyend+1],
                hx[1:pxstart,pystart:pyend+1],
                hx[1:pxstart,pystart-1:pyend],
                dt/eps_0/dx,dt/eps_0/dy, out=ez[1:pxstart,pystart:pyend+1])

            ez_op(ez[pxend+1:-1,pystart:pyend+1],
                hy[pxend+1:,pystart:pyend+1],
                hy[pxend:-1,pystart:pyend+1],
                hx[pxend+1:-1,pystart:pyend+1],
                hx[pxend+1:-1,pystart-1:pyend],
                dt/eps_0/dx,dt/eps_0/dy, out=ez[pxend+1:-1,pystart:pyend+1])  

            ez_op(ez[1:-1,1:pystart],
                  hy[1:,1:pystart],
                  hy[:-1,1:pystart],
                  hx[1:-1,1:pystart],
                  hx[1:-1,:pystart-1],
                  dt/eps_0/dx,dt/eps_0/dy, out=ez[1:-1,1:pystart])

            ez_op(ez[1:-1,pyend+1:-1],
                hy[1:,pyend+1:-1],
                hy[:-1,pyend+1:-1],
                hx[1:-1,pyend+1:],
                hx[1:-1,pyend:-1],
                dt/eps_0/dx,dt/eps_0/dy, out=ez[1:-1,pyend+1:-1])
            
            """E_z, J processing inside the plasma field"""
            ez_plasma_op(ez_0[pxstart:pxend+1,pystart:pyend+1],
                        hy[pxstart:pxend+1,pystart:pyend+1],
                        hy[pxstart-1:pxend,pystart:pyend+1],
                        hx[pxstart:pxend+1,pystart:pyend+1],
                        hx[pxstart:pxend+1,pystart-1:pyend],
                        D.real,B.real,F.real,j.real,1/(eps_0*dx),1/(eps_0*dy),1/eps_0,
                        out=ez[pxstart:pxend+1,pystart:pyend+1])
            
            j_plasma_op(j.real,D.real,F.real,nu,wp,C.real,B.real,
                        ez[pxstart:pxend+1,pystart:pyend+1],
                        ez_0[pxstart:pxend+1,pystart:pyend+1],
                        out=j)
            
        else:
            """E_z outside plasma field"""
            ez[1:pxstart,pystart:pyend+1] = ez[1:pxstart,pystart:pyend+1] + dt/(eps_0*eps_rel[1:pxstart,pystart:pyend+1]) * \
            ((hy[1:pxstart,pystart:pyend+1]-hy[:pxstart-1,pystart:pyend+1])/dx - \
            (hx[1:pxstart,pystart:pyend+1]-hx[1:pxstart,pystart-1:pyend])/dy)


            ez[pxend+1:-1,pystart:pyend+1] =  ez[pxend+1:-1,pystart:pyend+1] + dt/(eps_0*eps_rel[pxend+1:-1,pystart:pyend+1]) * \
            ((hy[pxend+1:,pystart:pyend+1]-hy[pxend:-1,pystart:pyend+1])/dx - \
            (hx[pxend+1:-1,pystart:pyend+1]-hx[pxend+1:-1,pystart-1:pyend])/dy)


            ez[1:-1,1:pystart] = ez[1:-1,1:pystart] + dt/(eps_0*eps_rel[1:-1,1:pystart]) * \
            ((hy[1:,1:pystart]-hy[:-1,1:pystart])/dx - \
            (hx[1:-1,1:pystart]-hx[1:-1,:pystart-1])/dy)
            

            ez[1:-1,pyend+1:-1] = ez[1:-1,pyend+1:-1] + dt/(eps_0*eps_rel[1:-1,pyend+1:-1]) * \
            ((hy[1:,pyend+1:-1]-hy[:-1,pyend+1:-1])/dx - \
            (hx[1:-1,pyend+1:]-hx[1:-1,pyend:-1])/dy)

            """E_z, J inside plasma field"""
            # Not updated with relative yet
            ez[pxstart:pxend+1,pystart:pyend+1] = D * ez_0[pxstart:pxend+1,pystart:pyend+1] + \
                B / (eps_0*dx) * (hy[pxstart:pxend+1,pystart:pyend+1] - hy[pxstart-1:pxend,pystart:pyend+1]) - \
                B / (eps_0*dy) * (hx[pxstart:pxend+1,pystart:pyend+1] - hx[pxstart:pxend+1,pystart-1:pyend]) - \
                F/eps_0*j


            j = j*(D-nu*F+wp**2*C*F/B) + (eps_0*wp**2*C/B)*ez[pxstart:pxend+1,pystart:pyend+1] + (eps_0*wp**2*F - eps_0*wp**2*C*D/B)*ez_0[pxstart:pxend+1,pystart:pyend+1]


        """Add input pulse"""
        if(n<last_input_timestep):
            for (row_slice, column_slice) in input_slices:
                ez[row_slice,column_slice] += input(t,dt)

        """Absorbing boundary conditions"""
        ez[0,:] = ez_0[1,:] +((c_0*dt-dx)/(c_0*dt+dx))*(ez[1,:]-ez_0[0,:])
        ez[:,0] = ez_0[:,1] +((c_0*dt-dy)/(c_0*dt+dy))*(ez[:,1]-ez_0[:,0])
        ez[nx,:]=ez_0[nx-1,:] +((c_0*dt-dx)/(c_0*dt+dx))*(ez[nx-1,:]-ez_0[nx,:])
        ez[:,ny]=ez_0[:,ny-1] +((c_0*dt-dy)/(c_0*dt+dy))*(ez[:,ny-1]-ez_0[:,ny])

        """Magnetic Field update equations"""
        if(USE_NUMBA):
            h_op(hy,ez[1:,:],ez[:-1,:],(dt/(mu_0*dx)),out=hy)
            h_op(hx,ez[:,1:],ez[:,:-1],(-dt/(mu_0*dy)),out=hx)
        else:
            hy = hy + (dt/(mu_0*mu_rel[1:,:]*dx))*(ez[1:,:]-ez[:-1,:])
            hx = hx - (dt/(mu_0*mu_rel[:,1:]*dy))*(ez[:,1:]-ez[:,:-1])

        """Poll & store requested locations"""
        for i, (row_slice, column_slice) in enumerate(poll_slices):
            polled_data[i][n] = ez[row_slice,column_slice]

        """Store ez at previous timestep"""
        ez_0 = ez.copy() 

    return polled_data, dt   
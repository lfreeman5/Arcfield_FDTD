import numpy as np
import matplotlib.pyplot as plt

MAX_FIGSIZE=10

def animateFDTD(n,nx,ny,dx,dy,ez,square_size=None):
    x = np.linspace(0,dx*nx,nx+1)
    y = np.linspace(0,dy*ny,ny+1)
    X,Y = np.meshgrid(x,y)
        
    aspect_ratio = np.ptp(y) / np.ptp(x)

    # Plot using pcolormesh
    # Create a figure and axis instance
    fig, ax = plt.subplots(figsize=(10,10))


    # Plot using pcolormesh with ax object
    pcm = ax.pcolormesh(X, Y, ez.T, shading='nearest',vmin=-20,vmax=20)  # Create a colormesh plot
    fig.colorbar(pcm, ax=ax)  # Add colorbar linked to the axis


    if(square_size is not None):
        half_size = square_size // 2
        cx, cy = len(X) // 2, len(Y) // 2

        coordinates = np.array([
            [x[cx-half_size], y[cy-half_size]],  # Bottom-left
            [x[cx+half_size], y[cy-half_size]],  # Bottom-right
            [x[cx+half_size], y[cy+half_size]],  # Top-right
            [x[cx-half_size], y[cy+half_size]],  # Top-left
            [x[cx-half_size], y[cy-half_size]]   # Close the square
        ])
        ax.plot(coordinates[:, 0], coordinates[:, 1], c='k', linewidth=2)


    ax.set_title(f'Electric Field Strength (Step {n}) - Relative Permittivity = 3000')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_aspect('equal', adjustable='box')
    plt.savefig(f'./plots/iter_{n}.png')
    plt.close(fig)
import matplotlib.pyplot as plt
import numpy as np

# 1. Grid Setup (from your initial code)
nx, ny, nz = 101, 101, 101
x = np.linspace(0, 10, nx)
y = np.linspace(0, 10, ny)
z = np.linspace(0, 10, nz)
dx = x[1] - x[0]  # Assumes dx = dy = dz = 0.1

# 2. Constants and Arrays
epsilon_0 = 8.854e-12
phi = np.zeros((nx, ny, nz))
rho = np.zeros((nx, ny, nz))

# Example: Place a point charge in the center of the 3D box
rho[50, 50, 50] = 1e-9

# 3. Professor's Relaxation Loop
n = 0
while n < 1000:
    phiold = phi.copy()  # Must use .copy() so phiold doesn't overwrite itself

    # Update all interior points [1:100] in x, y, and z simultaneously
    phi[1:100, 1:100, 1:100] = (
        rho[1:100, 1:100, 1:100] * dx**2 / epsilon_0
        + phiold[0:99, 1:100, 1:100]
        + phiold[2:101, 1:100, 1:100]
        + phiold[1:100, 0:99, 1:100]
        + phiold[1:100, 2:101, 1:100]
        + phiold[1:100, 1:100, 0:99]
        + phiold[1:100, 1:100, 2:101]
    ) / 6.0

    # Enforce grounded box walls (Dirichlet boundary conditions = 0)
    phi[0, :, :] = 0
    phi[100, :, :] = 0
    phi[:, 0, :] = 0
    phi[:, 100, :] = 0
    phi[:, :, 0] = 0
    phi[:, :, 100] = 0

    n = n + 1

# 4. Plot a 2D slice through the middle (z = 50)
plt.figure(figsize=(6, 5))
plt.imshow(phi[:, :, 50], origin='lower', extent=[0, 10, 0, 10], cmap='viridis')
plt.colorbar(label='Electric Potential $\phi$ (V)')
plt.title('3D Electrostatic Potential (Slice at z = 5)')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

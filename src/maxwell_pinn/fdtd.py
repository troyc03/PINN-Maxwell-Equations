import matplotlib.pyplot as plt
import numpy as np

# 1. Grid Setup (from your code)
nx, ny, nz = 101, 101, 101
x = np.linspace(0, 10, nx)
y = np.linspace(0, 10, ny)
z = np.linspace(0, 10, nz)
dx = x[1] - x[0]
dy = y[1] - y[0]
dz = z[1] - z[0]

# 2. Physical Constants & CFL Time Step
c = 299792458.0  # Speed of light (m/s)
mu0 = 4 * np.pi * 1e-7  # Permeability of free space
eps0 = 1 / (mu0 * c**2)  # Permittivity of free space

# Set dt to 99% of the 3D Courant stability limit
dt = 0.99 / (c * np.sqrt(1 / dx**2 + 1 / dy**2 + 1 / dz**2))

# 3. Initialize Field Arrays
Ex = np.zeros((nx, ny, nz))
Ey = np.zeros((nx, ny, nz))
Ez = np.zeros((nx, ny, nz))

Hx = np.zeros((nx, ny, nz))
Hy = np.zeros((nx, ny, nz))
Hz = np.zeros((nx, ny, nz))

# Precompute update coefficients
ce_x = dt / (eps0 * dx)
ce_y = dt / (eps0 * dy)
ce_z = dt / (eps0 * dz)

ch_x = dt / (mu0 * dx)
ch_y = dt / (mu0 * dy)
ch_z = dt / (mu0 * dz)

# 4. Main FDTD Time-Stepping Loop
num_steps = 120
center = nx // 2

for n in range(num_steps):
    # Update Magnetic Field (H) using curl of E
    Hx[:, :-1, :-1] -= ch_y * (Ez[:, 1:, :-1] - Ez[:, :-1, :-1]) - ch_z * (
        Ey[:, :-1, 1:] - Ey[:, :-1, :-1]
    )
    Hy[:-1, :, :-1] -= ch_z * (Ex[:-1, :, 1:] - Ex[:-1, :, :-1]) - ch_x * (
        Ez[1:, :, :-1] - Ez[:-1, :, :-1]
    )
    Hz[:-1, :-1, :] -= ch_x * (Ey[1:, :-1, :] - Ey[:-1, :-1, :]) - ch_y * (
        Ex[:-1, 1:, :] - Ex[:-1, :-1, :]
    )

    # Update Electric Field (E) using curl of H
    Ex[:, 1:, 1:] += ce_y * (Hz[:, 1:, 1:] - Hz[:, :-1, 1:]) - ce_z * (
        Hy[:, 1:, 1:] - Hy[:, 1:, :-1]
    )
    Ey[1:, :, 1:] += ce_z * (Hx[1:, :, 1:] - Hx[1:, :, :-1]) - ce_x * (
        Hz[1:, :, 1:] - Hz[:-1, :, 1:]
    )
    Ez[1:, 1:, :] += ce_x * (Hy[1:, 1:, :] - Hy[:-1, 1:, :]) - ce_y * (
        Hx[1:, 1:, :] - Hx[1:, :-1, :]
    )

    # Inject a soft Gaussian pulse at the center of Ez
    pulse = np.exp(-0.5 * ((n - 30) / 8.0) ** 2)
    Ez[center, center, center] += pulse

# 5. Visualize the XY center slice of Ez
plt.figure(figsize=(7, 6))
plt.imshow(
    Ez[:, :, center].T,
    origin='lower',
    extent=[0, 10, 0, 10],
    cmap='RdBu',
    vmin=-0.05,
    vmax=0.05,
)
plt.colorbar(label='Ez (V/m)')
plt.title(f'3D FDTD Wave Propagation (z = 5m, step = {num_steps})')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.tight_layout()
plt.show()

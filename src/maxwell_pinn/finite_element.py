import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


def generate_3d_tet_mesh(size=10.0, num_points=21):
    """Generates a 3D node list and tetrahedral elements from a structured grid."""
    coords = np.linspace(0, size, num_points)
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')
    nodes = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    # Helper to map (i, j, k) grid indices to a 1D node index
    def node_id(i, j, k):
        return i * (num_points**2) + j * num_points + k

    elements = []
    # Split each cube cell into 5 tetrahedra
    for i in range(num_points - 1):
        for j in range(num_points - 1):
            for k in range(num_points - 1):
                n000 = node_id(i, j, k)
                n100 = node_id(i + 1, j, k)
                n010 = node_id(i, j + 1, k)
                n110 = node_id(i + 1, j + 1, k)
                n001 = node_id(i, j, k + 1)
                n101 = node_id(i + 1, j, k + 1)
                n011 = node_id(i, j + 1, k + 1)
                n111 = node_id(i + 1, j + 1, k + 1)

                # Alternating parity keeps shared faces aligned across cubes
                if (i + j + k) % 2 == 0:
                    elements.extend([
                        [n000, n100, n010, n001],
                        [n100, n110, n010, n111],
                        [n010, n011, n001, n111],
                        [n100, n101, n001, n111],
                        [n100, n010, n001, n111],
                    ])
                else:
                    elements.extend([
                        [n100, n110, n000, n101],
                        [n000, n010, n110, n011],
                        [n000, n001, n101, n011],
                        [n110, n111, n101, n011],
                        [n000, n110, n101, n011],
                    ])

    return nodes, np.array(elements), coords


def solve_poisson_fem_3d(nodes, elements, size=10.0):
    """Assembles the global stiffness matrix K and solves K * phi = f."""
    epsilon_0 = 8.854e-12
    num_nodes = len(nodes)
    K = lil_matrix((num_nodes, num_nodes))
    f = np.zeros(num_nodes)

    # 1. Assemble Global Stiffness Matrix (K)
    for tet in elements:
        pts = nodes[tet]  # 4x3 matrix of vertex coordinates
        # Compute volume and shape function gradients using Jacobian
        J = np.column_stack([np.ones(4), pts])
        det_J = np.linalg.det(J)
        volume = abs(det_J) / 6.0

        # Gradients of linear basis functions (rows 1..3 of J^-1)
        grads = np.linalg.inv(J)[1:, :]  # 3x4 matrix

        # Local 4x4 stiffness matrix: K_local = volume * (grad_N^T * grad_N)
        K_local = volume * (grads.T @ grads)

        # Add local contributions into global sparse matrix
        for a in range(4):
            for b in range(4):
                K[tet[a], tet[b]] += K_local[a, b]

    # 2. Apply Point Charge Source at Center Node
    center_coord = np.array([size / 2, size / 2, size / 2])
    center_node = np.argmin(np.linalg.norm(nodes - center_coord, axis=1))
    f[center_node] = 1e-9 / epsilon_0

    # 3. Apply Grounded Boundary Conditions (phi = 0 on outer faces)
    tol = 1e-6
    boundary_mask = (
        (np.abs(nodes[:, 0]) < tol)
        | (np.abs(nodes[:, 0] - size) < tol)
        | (np.abs(nodes[:, 1]) < tol)
        | (np.abs(nodes[:, 1] - size) < tol)
        | (np.abs(nodes[:, 2]) < tol)
        | (np.abs(nodes[:, 2] - size) < tol)
    )

    K = K.tocsr()
    for idx in np.where(boundary_mask)[0]:
        K.data[K.indptr[idx] : K.indptr[idx + 1]] = 0.0
        K[idx, idx] = 1.0
        f[idx] = 0.0

    # 4. Solve Sparse Linear System
    phi = spsolve(K, f)
    return phi


if __name__ == '__main__':
    # Use N=21 for fast pure-Python assembly (N=101 has ~5 million tetrahedra)
    N = 21
    nodes, elements, coords = generate_3d_tet_mesh(size=10.0, num_points=N)
    phi_1d = solve_poisson_fem_3d(nodes, elements, size=10.0)

    # Reshape 1D node solution back into 3D grid for plotting
    phi_3d = phi_1d.reshape((N, N, N))
    center = N // 2

    plt.figure(figsize=(6, 5))
    plt.imshow(
        phi_3d[:, :, center].T, origin='lower', extent=[0, 10, 0, 10], cmap='viridis'
    )
    plt.colorbar(label='Electric Potential $\phi$ (V)')
    plt.title('3D FEM Electrostatic Potential (z = 5.0 m)')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.tight_layout()
    plt.show()

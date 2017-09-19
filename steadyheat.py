# TEST 3
# Solves steady state diffusion equation with finite volume method
# (Versteeg & Malaskera). Example 4.1.

import numpy as np
import matplotlib.pyplot as plt

# Parameters
nx = 10      # Node points
L = 0.5      # Length of object [Cm]
k = 1000     # Conductivity  [W/mK]
A = 10**-2   # Cross sectional area
T_A = 100.0  # Left Boundary Temperature [c]
T_B = 500.0  # Right Boundary Temperature [c]

# Dependant parameters
dx = (L/nx)   # Constant nodal spacing

# Grid generation for FV
centroid_arr = np.linspace(dx/2, L-dx/2, nx)

# Initilize solution vectors
M = np.zeros((nx, nx), dtype=float)
B = np.zeros((nx, 1), dtype=float)

# Interior Nodes
a_W = A*k/dx
a_E = A*k/dx
a_P = a_W + a_E

for i in range(1, nx - 1):
    M[i, i-1] = -  a_W
    M[i, i] = a_P
    M[i, i+1] = -a_E

# Left Boundary Node
a_W = 0
a_E = A*k/dx
S_P = -2*A*k/dx
S_u = (2*A*k/dx)*T_A
a_P = a_W + a_E - S_P

M[0, 0] = a_W + a_E - S_P
M[0, 1] = - a_E
B[0] = S_u


# Right Boundary Node
a_W = A*k/dx
a_E = 0
S_P = -2*A*k/dx
S_u = (2*A*k/dx)*T_B
a_P = a_W + a_E - S_P

M[-1, -1] = a_W + a_E - S_P
M[-1, -2] = - a_W
B[-1] = S_u

# Solve system
T = np.linalg.solve(M, B)

# Plot
plt.plot(centroid_arr, T, '.', label="Temperature Distribution")
plt.legend()
plt.show()

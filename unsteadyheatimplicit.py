# Explicit unsteady state diffusion equation with finite volume method
# (Versteeg & Malaskera). Example 8.1.
# Initial temp = 200c. Left side is insulated.
# @ t=0 right side is set at 0c.

import numpy as np
import matplotlib.pyplot as plt

# Parameters
nx = 5          # Num  of cells
time = 120.0      # final solution time
dt = 2.0        # [s]
L = 0.02        # Length of object [m]
k = 10.0        # Conductivity  [W/mK]
rhoc = 10 ** 7  # Rho*C
T_0 = 200.0     # Starting Temperature [c]
T_B = 0.0       # Right boundary temp [c]


# Dependant parameters
dx = (L/nx)   # Constant nodal spacing
nt = int(time/dt)
# Grid generation for FV
x_arr = np.linspace(dx/2, L-dx/2, nx)

# Initilize solution vectors
M = np.zeros((nx, nx), dtype=float)
B = np.zeros((nx, 1), dtype=float)
T = T_0*np.ones(nx)

for n in range(nt):
    Tn = T.copy()

    # Interior Nodes
    a_W = k/dx
    a_E = k/dx
    a_P0 = rhoc*dx/dt
    S_u = 0
    S_P = 0
    a_P = a_W + a_E + a_P0 - S_P
    
    for i in range(1, nx-1):
        M[i, i-1] = - a_W
        M[i, i] = a_P
        M[i, i+1] = - a_E
        B[i] = a_P0*Tn[i]
    
    # Left Node
    a_W = 0
    a_E = k/dx
    a_P0 = rhoc*dx/dt
    S_u = 0
    S_P = 0
    a_P = a_W + a_E + a_P0 - S_P
    M[0, 0] = a_P
    M[0, 1] = -a_E
    B[0] = a_P0 * Tn[0] + S_u

    
    # Right Node
    a_W = k/dx
    a_E = 0
    a_P0 = rhoc*dx/dt
    S_u = (2*k/dx)*T_B
    S_P = -2*k/dx
    a_P = a_W + a_E + a_P0 - S_P
    M[-1, -1] = a_P
    M[-1, -2] = -a_W
    B[-1] = a_P0 * Tn[-1] + S_u

    T = np.linalg.solve(M, B)
print(u)

# Plot
# plt.plot(centroid_arr, T, '.', label="Temperature Distribution")
# plt.legend()# right node
# plt.show()

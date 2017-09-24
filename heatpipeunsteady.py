# Explicit unsteady state diffusion equation with finite volume method
# (Versteeg & Malaskera)
# Left side: unit power. Right side: Convection

import numpy as np
import matplotlib.pyplot as plt

temp_array = []
# Parameters
nx = 30        # Num  of cells
time = 300    # final solution time
dt = 0.5       # [s]
L = 0.05       # Length of object [m]
k = 400        # Conductivity  [W/mK]
rho = 8940.0   # Density
C = 376.8      # Cp
T_0 = 23.0     # Starting Temperature [c]
Q_L = 15.0   # Unit Power
A = 0.01**2    # Cross sectional area
h = 15.0/(90-23)
T_amb = 23.0

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
    a_P0 = rho*C*dx/dt
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
    a_P0 = rho*C*dx/dt
    S_u = Q_L/A
    S_P = 0
    a_P = a_W + a_E + a_P0 - S_P
    M[0, 0] = a_P
    M[0, 1] = -a_E
    B[0] = a_P0 * Tn[0] + S_u

    # Right Node
    a_W = k/dx
    a_E = 0
    a_P0 = rho*C*dx/dt
    S_u = h*T_amb/A
    S_P = -h/A
    a_P = a_W + a_E + a_P0 - S_P
    M[-1, -1] = a_P
    M[-1, -2] = -a_W
    B[-1] = a_P0 * Tn[-1] + S_u

    T = np.linalg.solve(M, B)

    # Plotting Temperature
    temp_array.append(T[0])
#print(T)


# Plot Temperature over time
plt.subplot(111)
plt.plot(x_arr, T, '.')

# right node
plt.subplot(121)
plt.plot(dt*np.arange(nt), temp_array)
plt.show()
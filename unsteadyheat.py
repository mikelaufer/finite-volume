# Explicit unsteady state diffusion equation with finite volume method
# (Versteeg & Malaskera). Example 8.1.
# Initial temp = 200c. Left side is insulated.
# @ t=0 right side is set at 0c.

import numpy as np
import matplotlib.pyplot as plt

# Parameters
nx = 5          # Num  of cells
time = 4.0      # final solution time
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
u = T_0*np.ones(nx)

for n in range(nt):
    un = u.copy()

    # Interior Nodes
    a_W = k/dx
    a_E = k/dx
    a_P0 = rhoc*dx/dt
    a_P = rhoc*dx/dt
    S_u = 0
    # a_P = (a_W + a_E) + a_P0
    u[1:-1] = (a_W*un[0:-2] + a_E*un[2:] + (a_P0-(a_W+a_E))*un[1:-1] + S_u)/a_P

    # Left Node
    a_W = 0
    a_E = k/dx
    a_P0 = rhoc*dx/dt
    a_P = rhoc*dx/dt
    S_u = 0
    u[0] =  (a_E*un[1] + (a_P0-(a_W+a_E))*un[0])/a_P

    # right node
    a_W =  k/dx
    a_E = 0
    a_P0 = rhoc*dx/dt
    a_P = rhoc*dx/dt
    S_u = (2*k/dx)*(T_B-un[-1])
    u[-1] = (a_W*un[-2] + (a_P0-(a_W+a_E))*un[-1] + S_u)/a_P
    print(u)
    

# Plot
# plt.plot(centroid_arr, T, '.', label="Temperature Distribution")
# plt.legend()
# plt.show()

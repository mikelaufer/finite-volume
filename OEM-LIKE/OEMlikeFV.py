# Explicit unsteady state diffusion equation with finite volume method
# (Versteeg & Malaskera)
# Left side: unit power. Right side: Convection

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

# Parameters
nx = 120              # Num  of cells
dt = 0.2              # [s]
rho = 8940.0          # Density
C = 376.8             # Cp
T_0 = 23.0            # Starting Temperature [c]
T_amb = 23.0          # Ambient Temperature
Q_L = 15.0            # Just for hA_fin definition
hA_fin = Q_L/(90-23)  # Cconv

contact_dist = 0.005  # Contactor distance [m]
hp_dist = 0.05        # Heatpipe distance [m]
base_dist = 0.005     # Fin base dist [m]

# Dependant parameters
L = contact_dist + hp_dist + base_dist
dx = (L/nx)  

# Cross sectional area of cell
A1 = (0.01**2)*np.ones((int(contact_dist/dx),2))
A2 = (0.005**2)*np.ones((int(hp_dist/dx),2))
A2[0,0] = A1[-1,1]
A3 = (0.01**2)*np.ones((int(base_dist/dx),2))
A3[0,0] = A2[-1,1]
A = np.concatenate((A1, A2, A3), axis=0)

# Conductivity array
k = 400.0*np.ones(nx)
k[int(contact_dist/dx):int((contact_dist+hp_dist)/dx)] = 12000.0

# Grid generation for FV
x_arr = np.linspace(dx/2, L-dx/2, nx)

# Initilize solution vectors
M = np.zeros((nx, nx), dtype=float)
B = np.zeros((nx, 1), dtype=float)
a_W = np.zeros(nx)
a_E = np.zeros(nx)
a_P0 = np.zeros(nx)
S_u = np.zeros(nx)
S_P = np.zeros(nx)
a_P = np.zeros(nx)

# setup constant arrays
for i in range(nx):
    if i == 0:
        a_W[i] = 0
        a_E[i] = ((k[i]+k[i+1])/2)*A[i,1]/dx
        a_P0[i] = np.mean(A[i])*rho*C*dx/dt
        #S_u[i] = Q_L
        S_P[i] = 0
        a_P[i] = a_W[i] + a_E[i] + a_P0[i] - S_P[i]
    elif i == (nx-1):
        a_W[i] = ((k[i]+k[i-1])/2)*A[i,0]/dx
        a_E[i] = 0
        a_P0[i] = np.mean(A[i])*rho*C*dx/dt
        S_u[i] = hA_fin*T_amb
        S_P[i] = -hA_fin
        a_P[i] = a_W[i] + a_E[i] + a_P0[i] - S_P[i]
    else:
        a_W[i] = ((k[i]+k[i-1])/2)*A[i,0]/dx
        a_E[i] = ((k[i]+k[i+1])/2)*A[i,1]/dx
        a_P0[i] = np.mean(A[i])*rho*C*dx/dt
        S_u[i] = 0
        S_P[i] = 0
        a_P[i] = a_W[i] + a_E[i] + a_P0[i] - S_P[i]


def initT():
    T = T_0*np.ones(nx, dtype=float)
    return T


# Time stepping        
def FVtimestep(T, Q_L):
    # Left Node
    S_u[0] = Q_L
    M[0, 0] = a_P[0]
    M[0, 1] = -a_E[0]
    B[0] = a_P0[0] * T[0] + S_u[0]

    # Interior Nodes
    for i in range(1, nx-1):
        M[i, i-1] = -a_W[i]
        M[i, i] = a_P[i]
        M[i, i+1] = -a_E[i]
        B[i] = a_P0[i]*T[i] + S_u[i]

    # Right Node
    M[-1, -1] = a_P[-1]
    M[-1, -2] = -a_W[-1]
    B[-1] = a_P0[-1]*T[-1] + S_u[-1]
    T = np.linalg.solve(M, B)
    return T


if __name__ == "__main__":
    Q_L = 15.0           # Unit power
    time = 200           # Final solution time
    nt = int(time/dt)
    temp_array = []
    
    T = initT()
    for n in range(nt):
        T = FVtimestep(T, Q_L)
        temp_array.append(T[0])
        
    # Plot Temperature over time and x
    plt.figure(1)
    plt.plot(x_arr, T, '.')
    plt.title("Temperature vs distance")
    plt.xlabel("x [m]")
    plt.ylabel("Temperature [C]")

    plt.figure(2)
    plt.plot(dt*np.arange(nt), temp_array)
    plt.title("Temperature of Node1 vs time")
    plt.xlabel("Time [s]")
    plt.ylabel("Temperature [C]")
    plt.show()

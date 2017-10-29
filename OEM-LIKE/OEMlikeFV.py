# Implicit unsteady heat equation with finite volume method.
# Models an OEM style laptop cooler.
# Allows for changing local material properties and areas.
# (Versteeg & Malaskera)
# Left side: unit power. Right side: Convection

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded
import seaborn as sns
sns.set_style("whitegrid")

# Parameters
nx = 250              # Num  of cells
dt = 0.2              # [s]
T_0 = 30.0            # Starting Temperature [c]
T_amb = 23.0          # Ambient Temperature
Q_L = 15.0            # SS power, just for hA_fin definition

# Geometry
contact_dist = 0.005  # Contactor distance [m]
hp_dist = 0.03        # Heatpipe distance [m]
base_dist = 0.005     # Fin base dist [m]
fin_height = 0.01

L = contact_dist + hp_dist + base_dist + fin_height
dx = (L/nx)  

# Fin geometry and parameters
nfin = 8
h_fin = 100.0
t_fin = 0.0015
fin_width = 0.01
P_fin = nfin*(2*fin_width)  # total fin perimeter
A_fin = nfin*t_fin*fin_width

# Cross sectional area of cell
A_contact = 0.01**2
A_hp = 0.005**2
A_base = 0.01**2
A_fin = nfin*(t_fin*fin_width)

A1 = A_contact*np.ones((int(contact_dist/dx),2))
A2 = A_hp*np.ones((int(hp_dist/dx),2))
A2[0,0] = A1[-1,1]
A3 = A_base*np.ones((int(base_dist/dx),2))
A3[0,0] = A2[-1,1]
A4 = A_fin*np.ones((int(fin_height/dx),2))
A4[0,0] = A3[-1,1]
A = np.concatenate((A1, A2, A3, A4), axis=0)

# Material Properties
k_nom = 400.0
k_hp = 15000.0
C_nom = 376.8
C_hp = 376.8
rho_nom = 8940.0
rho_hp = 8940.0

k = k_nom*np.ones(nx)
k[int(contact_dist/dx):int((contact_dist+hp_dist)/dx)] = k_hp
C = C_nom**np.ones(nx)
C[int(contact_dist/dx):int((contact_dist+hp_dist)/dx)] = C_hp
rho = rho_nom*np.ones(nx)
rho[int(contact_dist/dx):int((contact_dist+hp_dist)/dx)] = rho_hp

# Grid generation for FV
x_arr = np.linspace(dx/2, L-dx/2, nx)

# Initilize solution vectors
M = np.zeros((nx, nx), dtype=float)
B = np.zeros(nx, dtype=float)
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
        a_P0[i] = np.mean(A[i])*rho[i]*C[i]*dx/dt
        #S_u[i] = Q_L
        S_P[i] = 0
        a_P[i] = a_W[i] + a_E[i] + a_P0[i] - S_P[i]
    elif i < int((contact_dist + hp_dist + base_dist)/dx) - 1:
        a_W[i] = ((k[i]+k[i-1])/2)*A[i,0]/dx
        a_E[i] = ((k[i]+k[i+1])/2)*A[i,1]/dx
        a_P0[i] = np.mean(A[i])*rho[i]*C[i]*dx/dt
        #S_u[i] = Q_L
        S_P[i] = 0
        a_P[i] = a_W[i] + a_E[i] + a_P0[i] - S_P[i]
    elif i == int((contact_dist + hp_dist + base_dist)/dx) - 1:
        a_W[i] = ((k[i]+k[i-1])/2)*A[i,0]/dx
        a_E[i] = ((k[i]+k[i+1])/2)*A[i,1]/dx
        a_P0[i] = np.mean(A[i])*rho[i]*C[i]*dx/dt
        S_u[i] = h_fin*(A_base-A_fin)*dx*T_amb
        S_P[i] = -h_fin*P_fin*dx
        a_P[i] = a_W[i] + a_E[i] + a_P0[i] - S_P[i]    
    elif i < nx-1:
        a_W[i] = ((k[i]+k[i-1])/2)*A[i,0]/dx
        a_E[i] = ((k[i]+k[i+1])/2)*A[i,1]/dx
        a_P0[i] = np.mean(A[i])*rho[i]*C[i]*dx/dt
        S_u[i] = h_fin*P_fin*dx*T_amb
        S_P[i] = -h_fin*P_fin*dx
        a_P[i] = a_W[i] + a_E[i] + a_P0[i] - S_P[i]
    else:
        a_W[i] = ((k[i]+k[i-1])/2)*A[i,0]/dx
        a_E[i] = 0
        a_P0[i] = np.mean(A[i])*rho[i]*C[i]*dx/dt
        S_u[i] = h_fin*P_fin*dx*T_amb
        S_P[i] = -h_fin*P_fin*dx
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

    # Solve tridiag system
    ud = np.insert(np.diag(M,1), 0, 0) # upper diagonal
    d = np.diag(M) # main diagonal
    ld = np.insert(np.diag(M,-1), len(B)-1, 0) # lower diagonal
    # simplified matrix
    ab = np.matrix([ud, d, ld])
    T = solve_banded((1, 1), ab, B)
    return T


if __name__ == "__main__":
    time = 300           # Final solution time
    nt = int(time/dt)
    temp_array = []
    
    T = initT()
    #Q_L1 = 20.0+2*np.random.random(200)-2*np.random.random(200)
    #Q_L2 = 15.0+2*np.random.random(nt-200+1)-2*np.random.random(nt-200+1)
    #Q_L = np.concatenate((Q_L1, Q_L2))

    # time1 = int(30/dt)
    # time2 = int(100/dt)
    # Q_L1 = 20.0+2*np.random.random(time1)-2*np.random.random(time1)
    # Q_L2 = 15.0+2*np.random.random(time2)-2*np.random.random(time2)
    # Q_L3 = 0.00+2*np.random.random(nt - time1 -time2 +1)
    # Q_L = np.concatenate((Q_L1, Q_L2, Q_L3))

    Q_L = 15*np.ones(nt+1)
    for n in range(nt):
        #Q_L = 15.0
        T = FVtimestep(T, Q_L[n])
        temp_array.append(T[0])
        
    # Plot Temperature over time and x
    plt.figure(1)
    plt.plot(x_arr, T, '.')
    plt.title("Temperature vs distance")
    plt.xlabel("x [m]")
    plt.ylabel("Temperature [C]")

    plt.figure(2)
    plt.plot(dt*np.arange(nt), temp_array, label="Temperature")
    plt.plot(dt*np.arange(nt+1), Q_L, label="Power", color="r")
    plt.legend()
    plt.title("Temperature and unit power vs time")
    plt.xlabel("Time [s]")
    plt.ylabel("Temperature [C] / Power[W]")
    plt.show()

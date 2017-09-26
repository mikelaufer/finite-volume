# Explicit unsteady state diffusion equation with finite volume method
# (Versteeg & Malaskera)
# Left side: unit power. Right side: Convection

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")


# Parameters
dt = 0.1              # [s]
T_0 = 23.0            # Starting Temperature [c]
T_amb = 23.0          # Ambient Temperature

A = 0.01**2           # Cross sectional area
L = 0.01              # Length of cooler (for comparison)
Q_L = 15.0            # Just for hA_fin definition
hA_fin = Q_L/(90-23)  # Cconv
rho = 8940.0          # Density
C = 376.8             # Cp

Cmass = rho*C*L*A     # Can be defined seperately
Cconv = hA_fin        # Can be defined seperately

def LCMtimestep(T, Q_L):
    T = T_amb + Q_L*(2*dt/(2*Cmass + dt*Cconv)) + (T-T_amb)*(2*Cmass -dt*Cconv)/(2*Cmass + dt*Cconv)
    return T


if __name__ == "__main__":
    Q_L = 15.0           # Unit power
    time = 200           # Final solution time
    nt = int(time/dt)
    temp_array = []
    T = T_0
    
    for n in range(nt):
        T = LCMtimestep(T, Q_L)
        temp_array.append(T)
        
    # Plot Temperature over time
    plt.figure(1)
    plt.plot(dt*np.arange(nt), temp_array)
    plt.title("Temperature vs Time")
    plt.xlabel("Time [s]")
    plt.ylabel("Temperature [C]")
    plt.show()

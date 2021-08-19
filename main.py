###############################################################################
# IMPORT MODULES
import numpy as np
import matplotlib.pyplot as plt

###############################################################################
# INPUTS

# Run Control
steps = 1400

# Constants
C = 0.5
gamma = 1.4

# 1D Grid
dx = 0.1
x = np.arange(0,3+dx,dx)
N = len(x)

# Geometry or Area Distribution
area = 1 + 2.2*(x - 1.5)**2

# Initial Conditions (ICs)
rho_ic = 1 - 0.3146*x
temp_ic = 1 - 0.2314*x
vel_ic = (0.1 + 1.09*x)*temp_ic**(1/2)

###############################################################################
# MAIN CODE

# Initialize global flow field variable data structures
rho = np.empty([steps,N])
vel = np.empty([steps,N])
temp = np.empty([steps,N])
pres = np.empty([steps,N])
mach = np.empty([steps,N])
mdot = np.empty([steps,N])

# Add ICs
rho[0] = rho_ic
vel[0] = vel_ic
temp[0] = temp_ic
pres[0] = rho_ic*temp_ic
mach[0] = vel_ic/np.sqrt(temp_ic)
mdot[0] = rho_ic*vel_ic*area


for step in range(1,steps):
    # Calculate time step, dt
    for i in range(1,N-1):
        dt_new = C * (dx/(temp[step-1, i]**(1/2)+vel[step-1, i]))
        if i == 1:
            dt = dt_new
        elif dt_new < dt:
            dt = dt_new

    # Initialize data arrays
    rho_new = np.empty([N])
    rho_bar = np.empty([N])
    drho_dt = np.empty([N])
    vel_new = np.empty([N])
    vel_bar = np.empty([N])
    dvel_dt = np.empty([N])
    temp_new = np.empty([N])
    temp_bar = np.empty([N])
    dtemp_dt = np.empty([N])
    # Perform predictor step on internal grid points
    for i in range(1,N-1):
        # Calculate predictor step derivatives
        drho_dt[i] = -rho[step-1, i]*(vel[step-1, i+1]-vel[step-1, i])/dx \
                   -rho[step-1, i]*vel[step-1, i]*(np.log(area[i+1])-np.log(area[i]))/dx \
                   -vel[step-1, i]*(rho[step-1, i+1]-rho[step-1, i])/dx
        dvel_dt[i] = -vel[step-1, i]*(vel[step-1, i+1]-vel[step-1, i])/dx \
                   -(1/gamma)*((temp[step-1, i+1]-temp[step-1, i])/dx \
                   + temp[step-1, i]/rho[step-1, i]*(rho[step-1, i+1]-rho[step-1, i])/dx)
        dtemp_dt[i] = -vel[step-1, i]*(temp[step-1, i+1]-temp[step-1, i])/dx \
                    - (gamma-1)*temp[step-1, i]*((vel[step-1, i+1]-vel[step-1, i])/dx \
                    + vel[step-1, i]*(np.log(area[i+1])-np.log(area[i]))/dx)
        # Calculate barred quantities
        rho_bar[i] = rho[step-1, i] + drho_dt[i]*dt
        vel_bar[i] = vel[step-1, i] + dvel_dt[i]*dt
        temp_bar[i] = temp[step-1, i] + dtemp_dt[i]*dt
    # Extrapolate to calculate inflow barred quantities
    rho_bar[0] = rho[step-1, 0]
    vel_bar[0] = 2*vel_bar[1] - vel_bar[2]
    temp_bar[0] = temp[step-1, 0]
    
    # Perform corrector step on interior grid points
    for i in range(1,N-1):
        # Calculate corrector step derivatives
        drhobar_dt = -rho_bar[i]*(vel_bar[i]-vel_bar[i-1])/dx \
                      -rho_bar[i]*vel_bar[i]*(np.log(area[i])-np.log(area[i-1]))/dx \
                      -vel_bar[i]*(rho_bar[i]-rho_bar[i-1])/dx
        dvelbar_dt = -vel_bar[i]*(vel_bar[i]-vel_bar[i-1])/dx \
                      - (1/gamma)*((temp_bar[i]-temp_bar[i-1])/dx \
                      + temp_bar[i]/rho_bar[i]*(rho_bar[i]-rho_bar[i-1])/dx)
        dtempbar_dt = -vel_bar[i]*(temp_bar[i]-temp_bar[i-1])/dx \
                      - (gamma-1)*temp_bar[i]*((vel_bar[i]-vel_bar[i-1])/dx \
                      + vel_bar[i]*(np.log(area[i])-np.log(area[i-1]))/dx)
        # Calculate average time derivatives
        drho_dt_av = 0.5 * (drho_dt[i] + drhobar_dt)
        dvel_dt_av = 0.5 * (dvel_dt[i] + dvelbar_dt)
        dtemp_dt_av = 0.5 * (dtemp_dt[i] + dtempbar_dt)
        # Calculate corrected flow field variables at next time step
        rho_new[i] = rho[step-1, i] + drho_dt_av*dt
        vel_new[i] = vel[step-1, i] + dvel_dt_av*dt
        temp_new[i] = temp[step-1, i] + dtemp_dt_av*dt

    # Calculate variables at inflow boundary
    rho_new[0] = 1
    vel_new[0] = 2*vel_new[1] - vel_new[2] 
    temp_new[0] = 1

    # Calculate variables at outflow boundary
    rho_new[-1] = 2*rho_new[-2] - rho_new[-3]
    vel_new[-1] = 2*vel_new[-2] - vel_new[-3]
    temp_new[-1] = 2*temp_new[-2] - temp_new[-3]

    # Add flow field variables at current time step to global arrays
    rho[step] = rho_new
    vel[step] = vel_new
    temp[step] = temp_new
    pres[step] = rho_new*temp_new
    mach[step] = vel_new/np.sqrt(temp_new)
    mdot[step] = rho_new*vel_new*area

###############################################################################
# PLOTTING
midpoint = int((N-1)/2)
fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1,sharex=True)

ax1.set(title='Timewise Variations of Flow Variables at the Nozzle Throat')
ax1.plot(range(steps),rho[:, midpoint])
ax1.set(ylabel=r'$\rho / \rho_0$')
ax1.set(ylim=(0.5,1.0))

ax2.plot(range(steps),temp[:, midpoint])
ax2.set(ylabel=r'$T/T_0$')
ax2.set(ylim=(0.6,1.0))

ax3.plot(range(steps),pres[:, midpoint])
ax3.set(ylabel=r'$p/p_0$')
ax3.set(ylim=(0.4,0.9))

ax4.plot(range(steps),mach[:, midpoint])
ax4.set(xlabel='Number of time steps',ylabel='M')
ax4.set(ylim=(0.9,1.3))

plt.show()
###############################################################################
# IMPORT PACKAGES
import numpy as np

###############################################################################
# INPUTS

# Constants
gamma = 1.4
C = 0.5

# 1D Grid
dx = 0.1
x = np.arange(0,3+dx,dx)

# Geometry or Area Distribution
area = 1 + 2.2*(x - 1.5)**2

# Initial Conditions (ICs)
rho_ic = 1 - 0.3146*x
temp_ic = 1 - 0.2314*x
vel_ic = (0.1 + 1.09*x)*temp_ic**(1/2)

###############################################################################
# FUNCTIONS



###############################################################################
# MAIN CODE

# Initialize flow field variable data structures and add ICs
rho = np.empty([1,len(x)])
vel = np.empty([1,len(x)])
temp = np.empty([1,len(x)])
rho[0] = rho_ic
vel[0] = vel_ic
temp[0] = temp_ic

# Iterate through internal grid points
#for i in x[1:len(x)-1]:
for i in range(1,len(x)):
    print(i)
    iplus = i + 1
    iminus = i - 1

    # Calculate predictor step derivatives
    drho_dt = -rho[-1, i]*(vel[-1, iplus]-vel[-1, i])/dx \
               -rho[-1, i]*vel[-1, i]*(np.log(area[iplus])-np.log(area[i]))/dx \
               -vel[-1, i]*(rho[-1, iplus]-rho[-1, i])/dx
    dvel_dt = -vel[-1, i]*(vel[-1, iplus]-vel[-1, i])/dx \
               -(1/gamma)*((temp[-1, iplus]-temp[-1, i])/dx \
               + temp[-1, i]/rho[-1, i]*(rho[-1, iplus]-rho[-1, i])/dx)
    dtemp_dt = -vel[-1, i]*(temp[-1, iplus]-temp[-1, i])/dx \
                - (gamma-1)*temp[-1, i]*((vel[-1, iplus]-temp[-1, i])/dx \
                + vel[-1, i]*(np.log(area[iplus])-np.log(area[i])/dx))

    # Calculate time step, dt
    print(temp[-1, i])
    print(vel[-1, i])
    dt = min(C * (dx/(temp[-1, i]**(1/2)+vel[-1, i])))

    # Calculate predicted or barred quantities
    rho_bar = rho[-1, i] + drho_dt*dt
    vel_bar = vel[-1, i] + dvel_dt*dt
    temp_bar = temp[-1, i] + dtemp_dt*dt
    
    # Calculate corrector step derivatives
    drhobar_dt = -rho_bar[i]*(vel_bar[i]-vel_bar[iminus])/dx \
                  -rho_bar[i]*vel_bar[i]*(np.log(area[i])-np.log(area[iminus]))/dx \
                  -vel_bar[i]*(rho_bar[i]-rho_bar[iminus])/dx
    dvelbar_dt = -vel_bar[i]*(vel_bar[i]-vel_bar[iminus])/dx \
                  - (1/gamma)*((temp_bar[i]-temp_bar[iminus])/dx \
                  + temp_bar[i]/rho_bar[i]*(rho_bar[i]-rho_bar[iminus])/dx)
    dtempbar_dt = -vel_bar[i]*(temp_bar[i]-temp_bar[iminus])/dx \
                  - (gamma-1)*temp_bar[i]*((vel_bar[i]-temp_bar[iminus])/dx \
                  + vel_bar[i]*(np.log(area[i])-np.log(area[iminus])/dx))

    # Calculate average time derivatives
    drho_dt_av = 0.5 * (drho_dt + drhobar_dt)
    dvel_dt_av = 0.5 * (dvel_dt + dvelbar_dt)
    dtemp_dt_av = 0.5 * (dtemp_dt + dtempbar_dt)

    # Calculate corrected flow field variables at next time step for all internal grid points
    rho_internal = rho[-1, i] + drho_dt_av*dt
    vel_internal = vel[-1, i] + dvel_dt_av*dt
    temp_internal = temp[-1, i] + dtemp_dt_av*dt
    
# Add in flow field variables at inflow boundary
rho_inflow = 1
rho_new = np.insert(rho_internal,rho_inflow)

vel_inflow = 2*vel_internal[0] - vel_internal[1]
vel_new = np.insert(vel_internal,vel_inflow)

temp_inflow = 1
temp_new = np.insert(temp_internal,temp_inflow)

# Add in flow field variables at outflow boundary
rho_outflow = 2*rho_internal[-1] - rho_internal[-2]
rho_new = np.append(rho_new,rho_outflow)

vel_outflow = 2*vel_internal[-1] - vel_internal[-2]
vel_new = np.append(vel_new,vel_outflow)

temp_outflow = 2*temp_internal[-1] - temp_internal[-2]
temp_new = np.append(temp_new,temp_outflow)

# Add flow field variables at current time step to array
rho = np.vstack((rho,rho_new))
vel = np.vstack((vel,vel_new))
temp = np.vstack((temp,temp_new))
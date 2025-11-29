"""
Ampliación de Matemáticas - Master Universitario en Sistemas Espaciales - ETSIAE
Milestone 4 : Linear problems. Regions of absolute stability.
1. Integrate the linear oscillator xdotdot+x = 0 with some initial conditions. Use
 Euler, Inverse Euler, Leap Frog, Crank Nicolson and fourth order Runge
 Kutta method.
 2. Regions of absolute stability of the above methods.
 3. Explain the numerical results based on regions of absolute stability.
"""
from numpy import array,concatenate,zeros,abs,max,log,real,imag,isclose,eye,block,all,meshgrid,linspace,finfo,copy
from numpy.linalg import norm,eigvals
import matplotlib.pyplot as plt

# Functions import
from Temporal_schemes import Euler, RK4, Crank_Nicolson, Inverse_Euler, Leap_Frog, RK45
from Cauchy_problem import Cauchy, Kepler_Force, F, Error_Richardson_Extrapolation, Convergence_Rate, Linear_oscillator
from Regions_of_stability import Region_of_stability

# ------------------------------------------------------------------
# Initial conditions for the linear oscillator problem
# ------------------------------------------------------------------
r0 = array([1, 0]) #Definition of initial position
dr0 = array([0, 1]) #Definition of initial velocity
U0=concatenate((r0,dr0)) #initial conditions U0
# Integration parameters
dt=0.1 #Definition of dt= t_total/n
n=500 #Definition of n number of steps
t_total = n*dt # total time
# t_total = 10.0
# dt = t_total/n

#------------------------------------------------------------------
# Cauchy problem integration with different temporal schemes
#------------------------------------------------------------------
U_Euler = Cauchy(Euler, Linear_oscillator, U0, t_total, n)
U_Inverse_euler = Cauchy(Inverse_Euler, Linear_oscillator, U0, t_total, n)
U_Leap_frog = Cauchy(Leap_Frog, Linear_oscillator, U0, t_total, n)
U_Crank_nicolson = Cauchy(Crank_Nicolson, Linear_oscillator, U0, t_total, n)
U_RK4 = Cauchy(RK4, Linear_oscillator, U0, t_total, n)
#U_RK45 = Cauchy(RK45, Linear_oscillator, U0, t_total, n)

#plotting results (2D graph)
plt.figure(figsize=(8, 6))
plt.plot(U_Euler[0,:], U_Euler[1,:], label='Euler')
plt.plot(U_Crank_nicolson[0,:], U_Crank_nicolson[1,:], label='Crank-Nicolson')
plt.plot(U_RK4[0,:], U_RK4[1,:], label='Runge-Kutta 4th Order')
plt.plot(U_Inverse_euler[0,:], U_Inverse_euler[1,:], label='Inverse Euler')
plt.plot(U_Leap_frog[0,:], U_Leap_frog[1,:], label='Leap Frog')
#plt.plot(U_RK45[0,:], U_RK45[1,:], label='Runge-Kutta 45')
plt.scatter(U_Euler[0,0], U_Euler[1,0], color='red', s=100, zorder=5, label='Initial position')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Oscillator')
plt.grid(True) 
plt.legend()  
plt.axis('equal') 
plt.show() 
# 3D GRAPH (Time as z axis)
t = U_Euler[4, :] 
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(U_Euler[0,:], U_Euler[1,:], U_Euler[4, :], label='Euler')
ax.plot(U_Crank_nicolson[0,:], U_Crank_nicolson[1,:], U_Crank_nicolson[4, :], label='Cranck-Nicolson')
ax.plot(U_RK4[0,:], U_RK4[1,:], U_RK4[4, :], label='Runge-Kutta 4th Order')
ax.plot(U_Inverse_euler[0,:], U_Inverse_euler[1,:], U_Inverse_euler[4, :], label='Inverse Euler')
ax.plot(U_Leap_frog[0,:], U_Leap_frog[1,:], U_Leap_frog[4, :], label='Leap Frog')
#ax.plot(U_RK45[0,:], U_RK45[1,:], U_RK45[-1, :], label='Runge-Kutta 45')
ax.scatter(U_Euler[0,0], U_Euler[1,0], U_Euler[4, 0], color='red', s=100, label='Initial position')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('t')
ax.set_title('Linear Oscillator')
ax.legend()
plt.show()

# ------------------------------------------------------------------
# Absolute stability regions
# ------------------------------------------------------------------

x = linspace(-4, 4, 600)
y = linspace(-4, 4, 600)
X, Y = meshgrid(x, y)
Z = X + 1j * Y

lim_euler = Region_of_stability(Euler, Z)
lim_rk4 = Region_of_stability(RK4, Z)
lim_inverse_euler = Region_of_stability(Inverse_Euler, Z)
lim_crank_nicolson = Region_of_stability(Crank_Nicolson, Z)
lim_leap_frog = Region_of_stability(Leap_Frog, Z)







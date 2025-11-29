"""
Ampliación de Matemáticas - Master Universitario en Sistemas Espaciales - ETSIAE
Milestone 2 : Prototypes to integrate orbits with functions.
1. Write a function called Euler to integrate one step. The function F(U, t)
of the Cauchy problem should be input argument.
2. Write a function called Crank_Nicolson to integrate one step.
3. Write a function called RK4 to integrate one step.
4. Write a function called Inverse_Euler to integrate one step.
5. Write a function to integrate a Cauchy problem. Temporal scheme, initial
condition and the function F(U, t) of the Cauchy problem should be input
arguments.
6. Write a function to express the force of the Kepler movement.
7. Integrate a Kepler with these latter schemes and explain the results.
8. Increase and decrease the time step and explained the results.
"""

from numpy import array,concatenate,zeros,abs,max
from numpy.linalg import norm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import warnings

# Functions import
from Temporal_schemes import Euler, RK4, Crank_Nicolson, Inverse_Euler, Leap_Frog
from Cauchy_problem import Cauchy, Kepler_Force, F 

# ------------------------------------------------------------------
# Intial conditions for the Kepler problem 
# ------------------------------------------------------------------
r0 = array([1, 0]) #Definition of initial position
dr0 = array([0, 1]) #Definition of initial velocity
U0=concatenate((r0,dr0)) #initial conditions U0 
# Integration parameters
dt=0.001 #Definition of dt= t_total/n
n=10000 #Definition of n number of steps
t_total = n*dt # total time
# t_total = 10.0
# dt = t_total/n


# ------------------------------------------------------------------
# Cauchy problem integration with different temporal schemes
# ------------------------------------------------------------------
U_Euler=Cauchy(Euler,Kepler_Force,U0,t_total,n)
U_Crank=Cauchy(Crank_Nicolson,Kepler_Force,U0,t_total,n) 
U_Runge=Cauchy(RK4,Kepler_Force,U0,t_total,n)
U_InverseEuler=Cauchy(Inverse_Euler,Kepler_Force,U0,t_total,n) #Needs lower dt to converge in the Kepler problem
U_LeapFrog=Cauchy(Leap_Frog,Kepler_Force,U0,t_total,n)

# ------------------------------------------------------------------
# Results plotting
# ------------------------------------------------------------------

# --- 2D GRAPH ---
plt.figure(figsize=(8, 6))
plt.plot( U_Euler[0, :], U_Euler[1, :], label='Euler')
plt.plot(U_Crank[0, :], U_Crank[1, :], label='Cranck-Nicolson')
plt.plot(U_Runge[0, :], U_Runge[1, :], label='Runge-Kutta 4th Order')
plt.plot(U_InverseEuler[0, :], U_InverseEuler[1, :], label='Inverse Euler')
plt.plot(U_LeapFrog[0, :], U_LeapFrog[1, :], label='Leap-Frog')
plt.scatter(U_Euler[0, 0], U_Euler[1, 0], color='red', s=100, zorder=5, label='Initial position')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Kepler orbit trajectory')
plt.grid(True) 
plt.legend()  
plt.axis('equal') 
plt.show() 

# --- 3D GRAPH ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(U_Euler[0, :], U_Euler[1, :], U_Euler[4, :] , label='Euler')
ax.plot(U_Crank[0, :], U_Crank[1, :], U_Crank[4, :] , label='Cranck-Nicolson')
ax.plot(U_Runge[0, :], U_Runge[1, :], U_Runge[4, :] , label='Runge-Kutta 4th Order')
ax.plot(U_InverseEuler[0, :], U_InverseEuler[1, :], U_InverseEuler[4, :] , label='Inverse Euler')
ax.plot(U_LeapFrog[0, :], U_LeapFrog[1, :], U_LeapFrog[4, :], label='Leap-Frog')
ax.scatter(U_Euler[0, 0], U_Euler[1, 0], U_Euler[4, 0] , color='red', s=100, label='Initial position')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('t')
ax.set_title('Kepler orbit trajectory with time')
ax.legend()
plt.show()
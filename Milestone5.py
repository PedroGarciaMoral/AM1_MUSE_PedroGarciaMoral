"""
Ampliación de Matemáticas - Master Universitario en Sistemas Espaciales - ETSIAE
Milestone 5 : N body problem.
 1. Write a function to integrate the N body problem.
 2. Simulate an example and discuss the results.
"""

from numpy import array,concatenate,zeros,abs,max,log,real,imag,meshgrid,linspace,sqrt
from numpy.linalg import norm,eigvals
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Functions import
from Temporal_schemes import Euler, RK4, Crank_Nicolson, Inverse_Euler, Leap_Frog
from Cauchy_problem import Cauchy, Kepler_Force, F, Error_Richardson_Extrapolation, Convergence_Rate, Linear_oscillator, N_Body_Problem_2D, N_Body_Problem_3D
from Regions_of_stability import Region_of_stability



# ---- Initial conditions for the solar system (2D) ----
# masses (solar masses)
masses = array([1.0, 1.0, 1.0])  # Sun, Earth, Jupiter
# initial positions
r0 = array([1.0, 0, -0.5, sqrt(3)/2, -0.5, -sqrt(3)/2, 0.0, 0.0])
# initial velocities
dr0 = array([0, 0.5, -sqrt(3)/4, -0.25,  sqrt(3)/4, -0.25, 0.0, 0.0])  
U0 = concatenate((r0, dr0))  # initial state vector

#iteration parameters
n = 2000        # steps
dt = 0.002      # years (~0.73 days)
t_total = n * dt
U = Cauchy(RK4, N_Body_Problem_2D, U0, t_total, n)

# --- 2D GRAPH ---
plt.figure(figsize=(8, 6))
plt.plot(U[0,:], U[1,:], label='Body 1')
plt.scatter(U[0,0], U[1,0], color='red', s=100, zorder=5, label='Initial position Body 1')
plt.plot(U[2,:], U[3,:], label='Body 2')
plt.scatter(U[2,0], U[3,0], color='red', s=100, zorder=5, label='Initial position Body 2')
plt.plot(U[4,:], U[5,:], label='Body 3')
plt.scatter(U[4,0], U[5,0], color='red', s=100, zorder=5, label='Initial position Body 3')
plt.plot(U[6,:], U[7,:], label='Body 4')
plt.scatter(U[6,0], U[7,0], color='red', s=100, zorder=5, label='Initial position Body 4')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('N Body Problem')
plt.grid(True) 
plt.legend()  
plt.axis('equal') 
plt.show() 

# --- 3D GRAPH ---
t = U[-1, :] 
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(U[0,:], U[1,:],t, label='Body 1')
ax.scatter(U[0,0], U[1,0],t[0], color='red', s=100, zorder=5, label='Initial position Body 1')
ax.plot(U[2,:], U[3,:],t, label='Body 2')
ax.scatter(U[2,0], U[3,0],t[0], color='red', s=100, zorder=5, label='Initial position Body 2')
ax.plot(U[4,:], U[5,:],t, label='Body 3')
ax.scatter(U[4,0], U[5,0],t[0], color='red', s=100, zorder=5, label='Initial position Body 3')
ax.plot(U[6,:], U[7,:],t, label='Body 4')
ax.scatter(U[6,0], U[7,0],t[0], color='red', s=100, zorder=5, label='Initial position Body 4')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('t')
ax.set_title('N Body Problem')
ax.legend()
plt.show()

# ---- Initial conditions for the N body problem in 3D ----
masses = array([1, 1, 1])  
# initial positions 
r0 = array([0.0, 0.0, 0.0,   
            1.0, 0.0, 0.0,
            2.0, 0.0, 0.0 ]) 
# initial velocities
dr0 = array([0.0, 0.0, 0.0,             
             0.0, 1.0, 0.0,
             0.0, 2.5, 0.0]) 
U0 = concatenate((r0, dr0))  # initial state vector
#iteration parameters
n = 2000        # steps
dt = 0.01      # years (~0.73 days)
t_total = n * dt

U = Cauchy(Euler, N_Body_Problem_3D, U0, t_total, n)

# --- 3D GRAPH ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(U[0,:], U[1,:], U[2,:], label='Body 1')
ax.scatter(U[0,0], U[1,0], U[2,0], color='red', s=100, label='Initial position Body 1')
ax.plot(U[3,:], U[4,:], U[5,:], label='Body 2')
ax.scatter(U[3,0], U[4,0], U[5,0], color='red', s=100, label='Initial position Body 2')
ax.plot(U[6,:], U[7,:], U[8,:], label='Body 3')
ax.scatter(U[6,0], U[7,0], U[8,0], color='red', s=100, label='Initial position Body 3')
# ax.plot(U[9,:], U[10,:], U[11,:], label='Body 4')
# ax.scatter(U[9,0], U[10,0], U[11,0], color='red', s=100, label='Initial position Body 4')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('N Body Problem')
ax.legend()
plt.show()
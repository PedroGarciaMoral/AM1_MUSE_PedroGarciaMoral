"""
Ampliación de Matemáticas - Master Universitario en Sistemas Espaciales - ETSIAE
Milestone 6: Lagrange points and their stability.
 1. Write a high order embedded Runge-Kutta method.
 2. Write function to simulate the circular restricted three body problem.
 3. Determination of the Lagrange points F(U) = 0.
 4. Stability of the Lagrange points: L1,L2,L3,L4,L5.
 5. Orbits around the Lagrange points by means of different temporal schemes.
"""

from numpy import array,concatenate,zeros,abs,max,log,real,imag,isclose,eye,block,all,meshgrid,linspace,finfo,copy,sqrt
from numpy.linalg import norm,eigvals
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import inspect

# Functions import
from Temporal_schemes import Euler, RK4, Crank_Nicolson, Inverse_Euler, Leap_Frog, RK45
from Cauchy_problem import Cauchy, Kepler_Force, F, Error_Richardson_Extrapolation,Convergence_Rate
from Cauchy_problem import Linear_oscillator, N_Body_Problem_2D, N_Body_Problem_3D, Circular_Restricted_3_Body_Problem_2D, Lagrange_Points
from Regions_of_stability import Region_of_stability

# ------------------------------------------------------------------
# Initial conditions for the linear oscillator problem
# ------------------------------------------------------------------
#r0 = array([0.75,0.0]) 
#dr0 = array([0.0, 0.6])
#r0 = array([-1.278,0.0]) 
#dr0 = array([0.0, 0.49])
r0 = array([0.487,0.86]) #Definition of initial position
dr0 = array([0, 0]) #Definition of initial velocity
U0=concatenate((r0,dr0)) #initial conditions U0
# Integration parameters
dt=0.01 #Definition of dt= t_total/n
n=8000 #Definition of n number of steps
t_total = n*dt # total time
# t_total = 10.0
# dt = t_total/n
mu=0.0121505837738  # Earth-Moon system mass parameter

U_RK4 = Cauchy(RK4, Circular_Restricted_3_Body_Problem_2D, U0, t_total, n)
#U_RK45 = Cauchy(RK45, Circular_Restricted_3_Body_Problem_2D, U0, t_total, n)
#U_Euler = Cauchy(Euler, Circular_Restricted_3_Body_Problem_2D, U0, t_total, n)
#U_Inverse_euler = Cauchy(Inverse_Euler, Circular_Restricted_3_Body_Problem_2D, U0, t_total, n)
#U_Crank_nicolson = Cauchy(Crank_Nicolson, Circular_Restricted_3_Body_Problem_2D, U0, t_total, n)


L_points=Lagrange_Points(mu)
L1_x, L1_y = L_points['L1']
L2_x, L2_y = L_points['L2']
L3_x, L3_y = L_points['L3']
L4_x, L4_y = L_points['L4']
L5_x, L5_y = L_points['L5']
print(f"L1: x = {L1_x:.6f}, y = {L1_y:.6f}")
print(f"L2: x = {L2_x:.6f}, y = {L2_y:.6f}")
print(f"L3: x = {L3_x:.6f}, y = {L3_y:.6f}")
print(f"L4: x = {L4_x:.6f}, y = {L4_y:.6f}")
print(f"L5: x = {L5_x:.6f}, y = {L5_y:.6f}")

plt.figure(figsize=(8, 6))
#plt.plot(U_Euler[0,:], U_Euler[1,:], label='Euler')
#plt.plot(U_Crank_nicolson[0,:], U_Crank_nicolson[1,:], label='Crank-Nicolson')
plt.plot(U_RK4[0,:], U_RK4[1,:], label='Runge-Kutta 4th Order')
#plt.plot(U_Inverse_euler[0,:], U_Inverse_euler[1,:], label='Inverse Euler')
#plt.plot(U_Leap_frog[0,:], U_Leap_frog[1,:], label='Leap Frog')
#plt.plot(U_RK45[0,:], U_RK45[1,:], label='Runge-Kutta 45')
plt.scatter(-mu,0, color='blue', s=100, zorder=5, label='Initial position Body 1 (Earth)')
plt.scatter(1-mu,0, color='black', s=100, zorder=5, label='Initial position Body 2 (Moon)')
plt.scatter(L1_x, L1_y, color='green', marker='D', s=50, zorder=6, label='$L_1$')
plt.scatter(L2_x, L2_y, color='green', marker='D', s=50, zorder=6, label='$L_2$')
plt.scatter(L3_x, L3_y, color='green', marker='D', s=50, zorder=6, label='$L_3$')
plt.scatter(L4_x, L4_y, color='purple', marker='*', s=150, zorder=6, label='$L_4$')
plt.scatter(L5_x, L5_y, color='purple', marker='*', s=150, zorder=6, label='$L_5$')
plt.scatter(U_RK4[0,0], U_RK4[1,0], color='red', s=100, zorder=5, label='Initial position')
plt.xlabel('X')
plt.ylabel('Y')
#limites del grafico [xmin,xmax,ymin,ymax]
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.title('Restricted Circular 3 Body Problem')
plt.grid(True) 
plt.legend()  
plt.axis('equal')    
plt.show() 




"""
Ampliación de Matemáticas - Master Universitario en Sistemas Espaciales - ETSIAE
Milestone 3 : Error estimation of numerical solutions.
 1. Write a function to evaluate errors of numerical integration by means of
 Richardson extrapolation. This function should be based on the Cauchy
 problem solution implemented in milestone 2.
 2. Numerical error or different temporal schemes: Euler, Inverse Euler, Crank
 Nicolson and fourth order Runge Kutta method.
 3. Write a function to evaluate the convergence rate of different temporal
 schemes.
 4. Convergence rate of the different methods with the time step.
"""

from numpy import array,concatenate,zeros,abs,max,log
from numpy.linalg import norm
import matplotlib.pyplot as plt

# Functions import
from Temporal_schemes import Euler, RK4, Crank_Nicolson, Inverse_Euler, Leap_Frog
from Cauchy_problem import Cauchy, Kepler_Force, F, Error_Richardson_Extrapolation, Convergence_Rate

# ------------------------------------------------------------------
# Initial conditions for the Kepler problem
# ------------------------------------------------------------------
r0 = array([1, 0]) #Definition of initial position
dr0 = array([0, 1]) #Definition of initial velocity
U0=concatenate((r0,dr0)) #initial conditions U0
# Integration parameters
dt=0.1 #Definition of dt= t_total/n
n=50 #Definition of n number of steps
t_total = n*dt # total time
# t_total = 10.0
# dt = t_total/n


# ------------------------------------------------------------------
# Richardson error extrapolation with different temporal schemes
# ------------------------------------------------------------------
Error_Richardson_Extrapolation_Euler=Error_Richardson_Extrapolation(Euler,Kepler_Force,U0,t_total, n)
Error_Richardson_Extrapolation_Crank=Error_Richardson_Extrapolation(Crank_Nicolson,Kepler_Force,U0,t_total, n)
Error_Richardson_Extrapolation_Runge=Error_Richardson_Extrapolation(RK4,Kepler_Force,U0,t_total, n)
Error_Richardson_Extrapolation_InverseEuler=Error_Richardson_Extrapolation(Inverse_Euler,Kepler_Force,U0,t_total, n) #Needs lower dt to converge in the Kepler problem

# ------------------------------------------------------------------
# Convergence rate with different temporal schemes
# ------------------------------------------------------------------
p_RK4=Convergence_Rate(RK4,RK4,Kepler_Force,U0,t_total,n)
print("p_RK4 = ",p_RK4)
p_Euler=Convergence_Rate(Euler,RK4,Kepler_Force,U0,t_total,n)
print("p_Euler = ",p_Euler)
p_InverseEuler=Convergence_Rate(Inverse_Euler,RK4,Kepler_Force,U0,t_total,n)
print("p_InverseEuler = ",p_InverseEuler)
p_Crank=Convergence_Rate(Crank_Nicolson,RK4,Kepler_Force,U0,t_total,n)
print("p_CrankNicolson = ",p_Crank)

# ------------------------------------------------------------------
# Results plotting
# ------------------------------------------------------------------

# --- 2D GRAPH ---
t1=Error_Richardson_Extrapolation_Euler[4, :]

plt.figure(figsize=(8, 6))
plt.plot(t1, Error_Richardson_Extrapolation_Euler[0, :], label='Euler')
plt.plot(t1, Error_Richardson_Extrapolation_Crank[0, :], label='Cranck-Nicolson')
plt.plot(t1, Error_Richardson_Extrapolation_InverseEuler[0, :], label='Inverse Euler')
plt.plot(t1, Error_Richardson_Extrapolation_Runge[0, :], label='Runge-Kutta 4th Order')
plt.xlabel('t')
plt.ylabel('X')
plt.title(f'Error with Richardson Extrapolation in x (with $dt$ = {dt:.4f})')
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(t1, Error_Richardson_Extrapolation_Euler[1, :], label='Euler')
plt.plot(t1, Error_Richardson_Extrapolation_Crank[1, :], label='Crank-Nicolson')
plt.plot(t1, Error_Richardson_Extrapolation_InverseEuler[1, :], label='Inverse Euler')
plt.plot(t1, Error_Richardson_Extrapolation_Runge[1, :], label='Runge-Kutta 4th Order')
plt.xlabel('t')
plt.ylabel('Y')
plt.title(f'Error with Richardson Extrapolation in y (with $dt$ = {dt:.4f})')
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(t1, Error_Richardson_Extrapolation_Euler[2, :], label='Euler')
plt.plot(t1, Error_Richardson_Extrapolation_Crank[2, :], label='Cranck-Nicolson')
plt.plot(t1, Error_Richardson_Extrapolation_InverseEuler[2, :], label='Inverse Euler')
plt.plot(t1, Error_Richardson_Extrapolation_Runge[2, :], label='Runge-Kutta 4th Order')
plt.xlabel('t')
plt.ylabel('dx/dt')
plt.title(f'Error with Richardson Extrapolation in dx/dt (with $dt$ = {dt:.4f})')
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(t1, Error_Richardson_Extrapolation_Euler[3, :], label='Euler')
plt.plot(t1, Error_Richardson_Extrapolation_Crank[3, :], label='Cranck-Nicolson')
plt.plot(t1, Error_Richardson_Extrapolation_InverseEuler[3, :], label='Inverse Euler')
plt.plot(t1, Error_Richardson_Extrapolation_Runge[3, :], label='Runge-Kutta 4th Order')
plt.xlabel('t')
plt.ylabel('dy/dt')
plt.title(f'Error with Richardson Extrapolation in dy/dt (with $dt$ = {dt:.4f})')
plt.grid(True)
plt.legend()
plt.show()


# --- Convergence rate vs number of steps (N) ---
# --- 2D GRAPH log(error)-log(Δt) ---
tiempo = 1.0
dts = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
Ns = [int(tiempo / dt) + 1 for dt in dts]
print("Ns = ", Ns)
err_euler = []
err_cn = []
err_inveuler = []
err_rk4 = []
for dt in dts:
     N = Ns[dts.index(dt)]
     U_ref = Cauchy(RK4,Kepler_Force, U0, tiempo, (N-1)*100+1)      # referencia muy fina
     U_euler = Cauchy(Euler,Kepler_Force, U0, tiempo, N)
     U_crank_nicolson = Cauchy(Crank_Nicolson,Kepler_Force, U0, tiempo, N)
     U_inverse_euler = Cauchy(Inverse_Euler,Kepler_Force, U0, tiempo, N)
     U_Runge = Cauchy(RK4,Kepler_Force, U0, tiempo, N)
     err_euler.append(norm(U_ref[0:4,-1] - U_euler[0:4,-1]))
     err_cn.append(norm(U_ref[0:4,-1] - U_crank_nicolson[0:4,-1]))
     err_inveuler.append(norm(U_ref[0:4,-1] - U_inverse_euler[0:4,-1]))
     err_rk4.append(norm(U_ref[0:4,-1] - U_Runge[0:4,-1]))
print("err_euler = ", err_euler)
plt.loglog(Ns, err_euler, 'o-', label='Euler (orden 1)')
plt.loglog(Ns, err_cn,    's-', label='Crank-Nicolson (orden 2)')
plt.loglog(Ns, err_inveuler, 'x-', label='Inverse Euler (orden 1)')
plt.loglog(Ns, err_rk4,   '^-', label='RK4 (orden 4)')
plt.xlabel('N')
plt.ylabel('Error en t= {tiempo:.1f})')
plt.legend()
plt.grid(True, which='both')
plt.title('Convergencia global de los métodos')
plt.show()

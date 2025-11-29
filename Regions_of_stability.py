""" Regions of stability of temporal schemes
----------------------------
Author: Pedro García Moral
Date: 2025 Q4
----------------------------

Contains:
- 1. Region of stability of different numerical schemes
- 2. Euler method stability region plot
- 3. Inverse euler method stability region plot
- 4. Cranck-Nicolson method stability region plot
- 5. RK4 method stability region plot
- 6. Leap Frog method stability region plot
----------------------------
"""
from numpy import array,concatenate,zeros,log, real, imag, linspace, meshgrid
from numpy.linalg import norm
import matplotlib.pyplot as plt
from Temporal_schemes import Euler, RK4, Crank_Nicolson, Inverse_Euler, Leap_Frog

"""
# ----------------------------------
1. Region of stability of different numerical schemes
# ----------------------------------
"""
def Region_of_stability(Temporal_Scheme, Z):
    if Temporal_Scheme == Euler:
        return R_euler(Z)
    elif Temporal_Scheme == Inverse_Euler:
        return R_inverse_euler(Z)
    elif Temporal_Scheme == Crank_Nicolson:
        return R_crack_nicolson(Z)
    elif Temporal_Scheme == RK4:
        return R_RK4(Z)
    elif Temporal_Scheme == Leap_Frog:
        return R_leap_frog(Z)
    else:
        raise ValueError("Temporal scheme not recognized for stability region calculation.")
    

"""
# ----------------------------------
2. Euler method stability region plot
# ----------------------------------
"""
def R_euler(z): 
    f = abs(1 + z)
    plt.figure(figsize=(10, 8))
    plt.contour(real(z), imag(z), f, levels=[0.2, 0.4, 0.6, 0.8, 1], colors='blue', linewidths=2)
    plt.title('Stability Region - Euler Method')
    plt.xlabel('Re(z)')
    plt.ylabel('Im(z)')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim(-4, 2)
    plt.ylim(-3, 3)
    plt.show()
    return f

"""
# ----------------------------------
3. Inverse euler method stability region plot
# ----------------------------------
"""
def R_inverse_euler(z):
    f = abs(1/(1 - z))
    plt.figure(figsize=(10, 8))
    plt.contour(real(z), imag(z), f, levels=[0.2, 0.4, 0.6, 0.8, 1], colors='green', linewidths=2, linestyles='--')
    plt.title('Stability Region - Inverse Euler Method (Region is OUTSIDE)')
    plt.xlabel('Re(z)')
    plt.ylabel('Im(z)')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim(-2, 4)
    plt.ylim(-3, 3)
    plt.show()
    return f

"""
# ----------------------------------
4. Cranck-Nicolson method stability region plot
# ----------------------------------
"""
def R_crack_nicolson(z):
    f = abs((1 + z / 2) / (1 - z / 2))
    plt.figure(figsize=(10, 8))
    plt.contour(real(z), imag(z), f, levels=[0.2, 0.4, 0.6, 0.8, 1], colors='orange', linewidths=2)
    plt.title('Stability Region - Crank-Nicolson Method')
    plt.xlabel('Re(z)')
    plt.ylabel('Im(z)')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim(-4, 2)
    plt.ylim(-3, 3)
    plt.show()
    return f

"""
# ----------------------------------
5. RK4 method stability region plot
# ----------------------------------
"""
def R_RK4(z):
    f = abs(1 + z + z**2 / 2 + z**3 / 6 + z**4 / 24)
    plt.figure(figsize=(10, 8))
    plt.contour(real(z), imag(z), f, levels=[0.2, 0.4, 0.6, 0.8, 1], colors='red', linewidths=2)
    plt.title('Stability Region - RK4 Method')
    plt.xlabel('Re(z)')
    plt.ylabel('Im(z)')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim(-4, 2)
    plt.ylim(-3, 3)
    plt.show()
    return f

"""
# ----------------------------------
6. Leap Frog method stability region plot
# ----------------------------------
"""
def R_leap_frog(z):
    plt.figure(figsize=(10, 8))
    plt.plot([0, 0], [-1, 1], color='purple', linewidth=4)
    plt.title('Stability Region - Leap Frog Method')
    plt.xlabel('Re(z)')
    plt.ylabel('Im(z)')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim(-4, 2)
    plt.ylim(-3, 3)
    plt.show()
    return (real(z) == 0) & (abs(imag(z)) <= 1)
""" Cauchy problem implementation
----------------------------
Author: Pedro García Moral
Date: 2025 Q4
----------------------------

Contains:
- 1. Cauchy problem integration with temporal schemes
- 2. Kepler force function
- 3. General function F (Kepler problem by default)
- 4. Error richardson extrapolation 
- 5. Error convergence rate evaluation
- 6. Linear oscillator function
- 7. N body problem in 2D function
- 8. N body problem in 3D function
- 9. Circular restricted 3 body problem in 2D function
- 10. Lagrange points for Circular restricted 3 body problem in 2D
----------------------------
"""
from numpy import array,concatenate,zeros,log, real, imag, linspace, meshgrid,sqrt
from numpy.linalg import norm
import matplotlib.pyplot as plt
from Temporal_schemes import Euler, RK4, Crank_Nicolson, Inverse_Euler, Leap_Frog
from scipy.optimize import fsolve
"""
----------------------------
1. FUNCTION CAUCHY PROBLEM
----------------------------
Function to integrate the Cauchy problem selecting the temporal scheme

"""
def Cauchy(Temporal_Scheme,F,U0,t_total,n):
    #bucle que recorre t cada dt y aplica el esquema temporal seleccionado
    U = zeros((len(U0)+1,n+1)) #Definition of the size of U state vector where each row is [U,t]'
    U[0:len(U0),0]= U0 #Include initial conditions in U
    dt=t_total/n 
    for i in range(n): #recorre desde i=0 hasta i=n-1
        U[:,i+1]=Temporal_Scheme(F, U[0:len(U0),i], U[len(U0),i], dt)
    return U

"""
----------------------------
2. FUNCTION KEPLER FORCE
----------------------------
Function to write the Kepler force instead of F. In this case the same as F by default
"""
def Kepler_Force(U, t): #Definition of F(dr,-r/norm(r)**3). Transforms R4 to R4
    dr = U[2:4] 
    r = U[0:2]  
    return concatenate((dr, -r / norm(r)**3))

"""
----------------------------
3. FUNCTION F
----------------------------
By default the kepler force function
"""
def F(dr,r): #Definition of F(dr,-r/norm(r)**3). Transforms R4 to R4
    F1=dr
    F2=-r/norm(r)**3
    F_resultante=concatenate((F1,F2))
    return F_resultante

"""
----------------------------
4. ERROR ESTIMATION USING RICHARDSON EXTRAPOLATION FUNCTION
----------------------------
Richardson extrapolation: phi = phi(h) + c1*h^p + c2*h^p... where phi is the exact solution, phi(h) is an aproximation using a step of size h, and c1,c2... are ctes unknown
The thing is to eliminate the terms with ctes using two different time steps: h and h/r
Finally you reach to: phi=(phi(h)-r^p*phi(h/r))/(1-r^p)

The ERROR with a step size h/r is defined as E(h/r) = phi - phi(h/r)
Introducing the Richarson extrapolation: E(h/r) = (phi(h)-r^p*phi(h/r))/(1-r^p) - phi(h/r)

The order of p for the different schemes:
- Euler/Inverse Euler: p=1
- Cranck-Nicolson/Leap-Frog: p=2
- RK4: p=4
"""
def Error_Richardson_Extrapolation(Temporal_Scheme,F,U0,t_total,n):
    dt = t_total/n
    r=2 # r is typically defined as 2 but could be another
    Error_richardson=zeros((5,(n-1)//r+2))
    phi_h=Cauchy(Temporal_Scheme,F,U0,t_total,(n-1)//r+1)         # Solution with step size  2*dt = dt'= t_total/((n-1)/r+1) 
    phi_h_r=Cauchy(Temporal_Scheme,F,U0,t_total, n)              # Solution with step size  dt = t_total/n
    phi_h_r_same_dimension=phi_h_r[:,::r]
    
    scheme_name = Temporal_Scheme.__name__
    if scheme_name == 'Euler':
        p = 1
    elif scheme_name == 'Inverse_Euler':
        p = 1
    elif scheme_name == 'Crank_Nicolson':
        p = 2
    elif scheme_name == 'Leap_Frog':
        p = 2
    elif scheme_name == 'RK4':
        p = 4
    else:
        raise ValueError(f"Esquema temporal '{scheme_name}' no reconocido.")

    Error_richardson[0:4,:]= (phi_h[0:4,:]-r**p*phi_h_r_same_dimension[0:4,:])/(1-r**p) - phi_h_r_same_dimension[0:4,:] 
    Error_richardson[4,:]= phi_h[4,:] # Time remains the same   
    return Error_richardson #Returns error estimation [5,n']

"""
----------------------------
5. CONVERGENCE RATE FUNCTION
----------------------------
Used to verify the scheme problem is being well calculated. It calculates the order p of the scheme that has to be:
- Euler/Inverse Euler: p=1
- Cranck-Nicolson: p=2
- RK4: p=4
The error E can be written in terms of the step size dt: E = C * delta^p where C is a constant
In terms of log: log E = p * log dt + log C which is a straight line
So if we calculate the Error at the same time T but using two different dt: E1, dt1, E2, dt2
p will be slope p=(log E2 - log E1)/(dt2 - dt1)
""" 
def Convergence_Rate(Temporal_Scheme,Temporal_Scheme_exact_solution,F,U0,t_total, n):
    dt = t_total/n
    U_exacto=Cauchy(Temporal_Scheme_exact_solution,F,U0,t_total,(n-1)*100+1) # High precision solution as exact solution
    U1=Cauchy(Temporal_Scheme,F,U0,t_total, (n-1)*2+1)
    U2=Cauchy(Temporal_Scheme,F,U0,t_total, (n-1)+1)
    E1=U_exacto[0:4,-1]-U1[0:4,-1] #-1 selecciona el ultimo elemento de la 5a columna
    E2=U_exacto[0:4,-1]-U2[0:4,-1]
    E1=norm(E1)
    E2=norm(E2)
    p=(log(E2) - log(E1))/(log(dt) - log(dt/2))
    # se podria añadir con una regresion de minimos cuadrados
    return p

"""

----------------------------
6. Linear oscillator function
----------------------------
Used to integrate the linear oscillator xdotdot + x = 0

"""
def Linear_oscillator(U, t):
    dr = U[len(U)//2:len(U)]
    r = U[0:len(U)//2]  
    return concatenate((dr, -r))

"""
----------------------------
7. N BODY PROBLEM (2D)
----------------------------
d2r_i/dt = + Sum(G * m_i * (r_j - r_i) / |r_j - r_i|^3)
 Sum for j inequal i to N
"""
def N_Body_Problem_2D(U,t):
    dr = U[len(U)//2:len(U)]
    r = U[0:len(U)//2]
    Nbodies = len(r)//2
    acc = zeros(len(r))
    for i in range(Nbodies):
        indice_ix = 2*i
        indice_iy = 2*i+1
        r_i = r[indice_ix:indice_iy+1] #[x_i,y_i]
        for j in range(Nbodies):
            if j != i:
                indice_jx = 2*j
                indice_jy = 2*j+1
                r_j = r[indice_jx:indice_jy+1] #[x_j,y_j]
                diff_r_ij = r_j - r_i
                dist_ij = norm(diff_r_ij)
                acc[indice_ix:indice_iy+1] = acc[indice_ix:indice_iy+1] + diff_r_ij / (dist_ij**3 + 1e-16)
    return concatenate((dr,acc))

"""
----------------------------
8. N BODY PROBLEM (3D)
----------------------------
d2r_i/dt = + Sum(G * m_i * (r_j - r_i) / |r_j - r_i|^3)
 Sum for j inequal i to N
"""
def N_Body_Problem_3D(U,t):
    dr = U[len(U)//2:len(U)]
    r = U[0:len(U)//2]
    Nbodies = len(r)//3
    acc = zeros(len(r))
    for i in range(Nbodies):
        indice_ix = 3*i
        indice_iy = 3*i+1
        indice_iz = 3*i+2
        r_i = r[indice_ix:indice_iz+1] #[x_i,y_i,z_i]
        for j in range(Nbodies):
            if j != i:
                indice_jx = 3*j
                indice_jy = 3*j+1
                indice_jz = 3*j+2
                r_j = r[indice_jx:indice_jz+1] #[x_j,y_j,z_j]
                diff_r_ij = r_j - r_i
                dist_ij = norm(diff_r_ij)
                acc[indice_ix:indice_iz+1] = acc[indice_ix:indice_iz+1] + diff_r_ij / (dist_ij**3 + 1e-16)
    return concatenate((dr,acc))

"""
----------------------------
9. CIRCULAR RESTRICTED 3 BODY PROBLEM (2D)
----------------------------
General N body problem
d2r_i/dt = - Sum(G * m_i * (r_j - r_i) / |r_j - r_i|^3)
 Sum for j inequal i to N
2 big masses M1,M2 and the third one is very small m3=0. With this assumption, M1,M2 will have a Keplerian 2 body orbit
Adimensional units: - M1+M2=1
                    - Distance between M1 and M2 is R=1
                    - Gravitatory Constant G=1
                    - Angular velocity constant (for circular orbit) w=1
Mass parameter mu=M2/(M1+M2); M1=1-mu; M2=mu
Reference system rotates with M1,M2 with w. M1 and M2 are fixed in this reference system. M1(-mu,0), M2(1-mu,0). The origin is the CM
The problem will study the movement of m3. The equation:
d2r/dt = - grad(pot) - 2w x dr/dt where - 2w x dr/dt is the Coriolis Force. Fx (circular) = 2*dy/dt; Fy (circular) = -2*dx/dt
and pot= pot_grav + pot_centrif = -GM1/sqrt((x-x1)^2+y^2) -GM2/sqrt((x-x2)^2+y^2) -1/2(x^2+y^2)= -(1-mu)/sqrt((x+mu)^2+y^2) -mu/sqrt((x-1+mu)^2+y^2) -1/2(x^2+y^2)
Then d2x/dt= 2*dy/dt - d(pot)/dx = 2*dy/dt + x -(1-mu)*(x+mu)/sqrt((x+mu)^2+y^2)^3 -mu*(x-1+mu)/sqrt((x-1+mu)^2+y^2)^3 
     d2y/dt= -2*dx/dt - d(pot)/dx = -2*dx/dt + y -(1-mu)*y/sqrt((x+mu)^2+y^2)^3 -mu*y/sqrt((x-1+mu)^2+y^2)^3
"""
def Circular_Restricted_3_Body_Problem_2D(U,t): 
    mu=0.01215058 #Earth-Moon
    dr = U[len(U)//2:len(U)]
    r = U[0:len(U)//2]
    F2=zeros((2))
    F2[0]=2*dr[1] + r[0] -(1-mu)*(r[0]+mu)/sqrt((r[0]+mu)**2+r[1]**2)**3 -mu*(r[0]-1+mu)/sqrt((r[0]-1+mu)**2+r[1]**2)**3 
    F2[1]=-2*dr[0] + r[1] -(1-mu)*r[1]/sqrt((r[0]+mu)**2+r[1]**2)**3 -mu*r[1]/sqrt((r[0]-1+mu)**2+r[1]**2)**3
    return concatenate((dr,F2))

"""
----------------------------
10. LAGRANGE POINTS FOR CIRCULAR RESTRICTED 3 BODY PROBLEM
----------------------------
d2x/dt=dx/dt=0
d2x/dt=dx/dt=0

Then: x -(1-mu)*(x+mu)/sqrt((x+mu)^2+y^2)^3 -mu*(x-1+mu)/sqrt((x-1+mu)^2+y^2)^3=0
      y -(1-mu)*y/sqrt((x+mu)^2+y^2)^3 -mu*y/sqrt((x-1+mu)^2+y^2)^3=0
"""
def Lagrange_Points(mu):
    def Equation(x):
        return x - (1 - mu) * (x + mu) / (x + mu)**3 - mu * (x - 1 + mu) / (x - 1 + mu)**3
    x_L1_guess = 1 - mu - 0.1 
    L1_x = fsolve(Equation, x_L1_guess)[0]
    x_L2_guess = 1 - mu + 0.1 
    L2_x = fsolve(Equation, x_L2_guess)[0]
    x_L3_guess = -mu - 1.05 
    L3_x = fsolve(Equation, x_L3_guess)[0]
    
    x_L45 = 0.5 - mu
    y_L4 = sqrt(3) / 2
    y_L5 = -sqrt(3) / 2
    
    L_points = {
        'L1': (L1_x, 0.0),
        'L2': (L2_x, 0.0),
        'L3': (L3_x, 0.0),
        'L4': (x_L45, y_L4),
        'L5': (x_L45, y_L5),
    }
    
    return L_points
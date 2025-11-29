""" Temporal schemes implementation
----------------------------
Author: Pedro García Moral
Date: 2025 Q4
----------------------------

Contains:
- 1. Euler scheme
- 2. 1st order Cranck-Nicolson scheme
- 3. 4th order explicit Runge-Kutta scheme
- 4. Inverse Euler scheme
- 5. Leap Frog scheme
- 6. Embedded Runge-Kutta45 with adaptative time step
----------------------------
"""
from numpy import array,concatenate,zeros,abs,max, sqrt
from numpy.linalg import norm

"""
----------------------------
1. FUNCTION EULER SCHEME
----------------------------
Euler scheme has an spectral radius p>1 so is unstable
Explicit method U_n+1 = U_n + dt * F(dr_n,r_n)
"""
def Euler(F,U,t,dt):
    return concatenate((U + dt * F(U, t), [t + dt])) 

"""
----------------------------
2. FUNCTION 1st ORDER CRANCK-NICOLSON SCHEME
----------------------------
Implicit method U_n+1 = U_n + dt/2 * (F(U_n,t) + F(U_n+1,t+dt))

To solve the implicit method, first start with an Euler scheme U_n+1 = U_n + dt * F(U_n) and
introduce it in the Cranck_Nicolson scheme U_n+1 = U_n + dt/2 * (F(U_n) + F(U_n+1)) to solve 
it iteratively until convergence.
"""
def Crank_Nicolson(F,U,t,dt):
    y = U + dt * F(U,t)   # 1st iteration with Euler scheme
    tol = 1.0
    while abs(tol)>1e-6:
      y_previous = y #previous value of y to calculate the error
      y = U + dt/2 * (F(U,t)+F(y,t+dt)) #Cranck-Nicolson scheme
      tol=norm(y-y_previous) # Calculates the error
    return concatenate((y,[t+dt]))

"""
----------------------------
3. FUNCTION 4th ORDER EXPLICIT RUNGE-KUTTA SCHEME
----------------------------
Explicit method U_n+1 = U_n + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
where   k1 = F(U0, t)
        k2 = F(U0 + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = F(U0 + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = F(U0 + dt * k3, t + dt)
"""
def RK4(F,U,t,dt):
    k1 = F(U, t)
    k2 = F(U + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = F(U + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = F(U + dt * k3, t + dt)
    return concatenate((U + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4), [t + dt]))

"""
----------------------------
4. FUNCTION INVERSE EULER SCHEME
----------------------------
Implicit method U_n+1 = U_n + dt* (F(U_n+1,t+dt))
To solve the implicit method, first start with an Euler scheme U_n+1= U_n + dt * F(U_n) and
introduce it in the Inverse Euler scheme U_n+1= U_n + dt * (F(U_n+1)) 
"""
def Inverse_Euler(F,U,t,dt):
    y = U + dt * F(U,t)   # 1st iteration with Euler scheme
    tol = 1.0
    while abs(tol)>1e-6:
      y_previous = y #previous value of y to calculate the error
      y = U + dt * F(y,t+dt) #Inverse-Euler scheme
      tol=norm(y-y_previous) # Calculates the error
    return concatenate((y,[t+dt]))

"""
----------------------------
5. LEAP FROG FUNCTION
----------------------------
Leap Frog scheme: U_n+1 = U_n-1 + 2* delta_T * F(U_n,t)
Multi-step scheme that needs to start with another scheme. Euler scheme is often used for the first step
"""
def Leap_Frog(F,U,t,dt):
    if t==0:
        U_new = Euler(F,U,t,dt)
    else:
        N = len(U)
        a_i = F(U, t)[N//2:N]
        v_i_plus_one_half = U[N//2:N] + 0.5*a_i*dt
        x_i_plus_one = U[0:N//2] + v_i_plus_one_half*dt
        a_i_plus_one = F(concatenate((x_i_plus_one,zeros(N//2))),t+dt)[N//2:N]
        v_i_plus_one = v_i_plus_one_half + 0.5*a_i_plus_one*dt
        U_new = concatenate((x_i_plus_one, v_i_plus_one, [t + dt]))
    return U_new

"""
----------------------------
6. EMBEDDED RUNGE-KUTTA SCHEME RK45
----------------------------
EMBEDDED RUNGE-KUTTA SCHEME RK45 with adaptative time step
        k1 = dt*F(U_n, t)
        k2 = dt*F(U_n + 1/4*k1, t_n + dt/4)
        k3 = dt*F(U_n + 3/32*k1 + 9/32*k2, t_n + dt*3/8)
        k4 = dt*F(U_n + 1932/2197*k1 - 7200/2197*k2 + 7296/2197*k3, t_n + dt*12/13)
        k5 = dt*F(U_n + 439/216*k1 - 8*k2 + 3680/513*k3 - 845/4104*k4, t_n + dt)
        k6 = dt*F(U_n - 8/27*k1 + 2*k2 - 3544/2565*k3 + 1859/4104*k4 - 11/40*k5, t_n + dt/2)
        Order 4 estimation--> Uo4_n+1 = U_n + (k1*25/216 + 0*k2 + 1408/2565*k3 + 2197/4104*k4 - 1/5*k5)
        Order 5 estimation--> Uo5_n+1 = U_n + (k1*16/135 + 0*k2 + 6656/12825*k3 + 28561/56430*k4 - 9/50*k5 + 2/55*k6)
        Error_array=U_o5_n+1-U_o4_n+1 
        Tol=Tol_a+Tol_r*norm(U_n)
        E = sqrt (1/len(U0) * Sum (Error_array(j)/Tol(j)))
        S=(1/E)^(1/5) where 5 is p+1 of the lower order method
        if E<=1: Integration continues (dt is accepted)
            U_n+1=Uo5_n+1
        elif E>1: Integration is repeated (dt is reduced)
            dt=dt*S
            i=i-1
"""
def RK45(F, U0, t, dt):
    dim = len(U0)
    U = zeros((dim+1, 2))
    U[0:dim, 0] = U0
    Tol_a = 1e-10
    Tol_r = 1e-10
    dt_attempt = dt
    while True:
        k1 = dt_attempt * F(U0, t)
        k2 = dt_attempt * F(U0 + 1/4*k1,                         t + dt_attempt/4)
        k3 = dt_attempt * F(U0 + 3/32*k1 + 9/32*k2,              t + dt_attempt*3/8)
        k4 = dt_attempt * F(U0 + 1932/2197*k1 - 7200/2197*k2 
                                 + 7296/2197*k3,                 t + dt_attempt*12/13)
        k5 = dt_attempt * F(U0 + 439/216*k1 - 8*k2 
                                 + 3680/513*k3 - 845/4104*k4,    t + dt_attempt)
        k6 = dt_attempt * F(U0 - 8/27*k1 + 2*k2 
                                 - 3544/2565*k3 + 1859/4104*k4 
                                 - 11/40*k5,                     t + dt_attempt/2)
        U4 = U0 + (25/216)*k1 + (1408/2565)*k3 + (2197/4104)*k4 - (1/5)*k5
        U5 = U0 + (16/135)*k1 + (6656/12825)*k3 + (28561/56430)*k4 - (9/50)*k5 + (2/55)*k6
        Error_array = U5 - U4
        Tol = Tol_a + Tol_r * abs(U0)
        Ej_Tolj = 0.0
        for j in range(dim):
            Ej_Tolj = Ej_Tolj + (Error_array[j] / Tol[j])**2
        E = sqrt(Ej_Tolj / dim)
        if E <= 1.0:
            break   # paso aceptado
        S = (1/E)**(1/5)
        dt_attempt = dt_attempt*S   # paso rechazado; repetir
    # # paso aceptado: actualizar estado y tiempo
    S = (1/E)**(1/5)
    dt_new = dt_attempt * S
    #print(f"dt updated from {dt} to {dt_new}")
    U[0:dim, 1] = U5
    U[dim, 1] = t + dt_new
    return U[:, 1]

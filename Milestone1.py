#Milestone 1 - AM1- MUSE
#Integracion numérica sin funciones
#Este código integra la órbita de un cuerpo bajo la influencia de un campo gravitatorio central 
#Se utilizan los métodos de Euler, RK4 y Crank-Nicolson
#Como se ha evitado el uso de funciones, para cambiar de sistema numérico habra que comentar/descomentar las lineas correspondientes

from numpy import array, concatenate, zeros  
from numpy.linalg import norm   
import matplotlib.pyplot as plt

# Función del sistema
def F(U):
    r = U[0:2]
    dr = U[2:4]
    return concatenate((dr, -r/norm(r)**3))

# Condiciones iniciales
r0 = array([0.0, 1.0])
dr0 = array([1.0, 0.0])
U0 = concatenate((r0, dr0))   # [x, y, vx, vy]

# Discretización temporal
T = 15
N = 100
dt = T / N

# Inicialización del array de soluciones
U = zeros((N+1, 4))
U[0, :] = U0

# Iteración método Crank-Nicolson
for n in range(0,N):
    #EULER
    #txt = f"Euler"
    #U[n+1]=U[n]+dt*F(U[n])

    #RK4
    #txt = f"RK4"
    #k1 = F(U[n,:])
    #k2 = F(U[n,:] + 0.5 * dt * k1)
    #k3 = F(U[n,:] + 0.5 * dt * k2)
    #k4 = F(U[n,:] + dt * k3)
    #U[n+1,:] = U[n,:] + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

    #CN
    txt = f"Crank-Nicolson"
    itU = U[n] + dt * F(U[n])   # 1a aproximación de U[n+1] con Euler
    for it in range(100):  # 10 iteraciones para mejorar la aproximación
        U_new = U[n, :] + (dt/2) * (F(U[n, :]) + F(itU))
        if norm(U_new - itU) < 1e-15:
            U[n+1, :] = U_new
            #print(f"Convergencia alcanzada en paso {n}")
            break
        itU = U_new
    else:
        U[n+1, :] = itU
        print(f"No convergencia en paso {n}, error {norm(U[n+1,:]-itU)}")

# Graficar trayectoria
x = U[:,0]
y = U[:,1]

plt.plot(x, y)
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.title("Órbita con " + txt)
plt.show()

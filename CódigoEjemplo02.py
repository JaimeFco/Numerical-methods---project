
# coding: utf-8

# ## Código Ejemplo 2

# In[1]:


import autograd.numpy as np
from autograd import grad 
import autograd.numpy.random as npr
from autograd.core import primitive
from matplotlib import pyplot as plt


# In[2]:


nx = 10        # Numero de puntos
dx = 2. / nx   # Espacio entre puntos


# In[3]:


# Función sigmoide
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Derivada del sigmoide con respecto x
def sigmoid_grad(x):
    return sigmoid(x) * (1 - sigmoid(x))


# Red neuronal con tres neuronas de pesos W
def neuralNetwork(W, x):
    a1 = sigmoid(np.dot(x, W[0])+W[2])
    return np.dot(a1, W[1])


# Derivada de la red neuronal con respecto de x con pesos W
def d_neuralNetwork_dx(W, x, k=1):
    return np.dot(sigmoid_grad(np.dot(x, W[0])+W[2]), W[1]*W[0].T)


# Función a minimizar con pesos W, xi's en x
def lossFunction(W, x):
    loss_sum = 0.
    for xi in x:
        net_out = neuralNetwork(W, xi)[0][0]       # Valor de la red neuronal en xi
        psy_t = 0. + xi * net_out                   # trial solution
        d_net_out = d_neuralNetwork_dx(W, xi)[0][0]# derivada de la red neuronal evaluada en xi
        d_psy_t = net_out + xi * d_net_out          # derivada de trial solution
        func = F(xi, psy_t)                         # sumando izquierdo de la cosa a minimizar
        err_sqr = (d_psy_t - func)**2               # Saca cuadrado

        loss_sum += err_sqr                         # Suma de los cuadrados
    return loss_sum                                 # Regresa valor de E[x, W]


# F
def F(x, psy):
    return B(x) - psy * A(x)


# Parte dependiente de psi
def A(x):
    return 1./5.


# Parte independiente de psi
def B(x):
    return np.exp(-x/5.)*np.cos(x)


# Pesos en la red neuronal, dos vectores de 1x12 y 12x1 (De input a hidden layer, de hidden layer a output), Biasis
# Número de neuronas: 
N = 3
W = [npr.randn(1, N), npr.randn(N, 1), npr.randn(1, N)]


# In[5]:


x_space = np.linspace(0, 2, nx)   # Creamos 10 puntos equiespaciados en [0, 2]


lmb = 0.01

## Entrenando la red neuronal

for i in range(1000):
    loss_grad =  grad(lossFunction)(W, x_space)   # Arreglo n-dimensional
    
    # Restandole el gradiente
    W[0] = W[0] - lmb * loss_grad[0]
    W[1] = W[1] - lmb * loss_grad[1]
    W[2] = W[2] - lmb * loss_grad[2]
    
    
    if lossFunction(W, x_space) < 0.01:
        print ("lossFunction = ",lossFunction(W, x_space), ", Iter: ", i)
        break
print ("lossFunction = ",lossFunction(W, x_space))


# In[6]:


Psi_NN = [ xi * neuralNetwork(W, xi)[0][0] for xi in np.linspace(0, 2, nx)] # Guarda los resultados en los puntos xi

# Imprime
plt.figure()
plt.plot(np.linspace(0, 2, nx), Psi_NN, "red")


def psy_analytic(x):
    return np.exp(-x/5.)*np.sin(x)


y_space = psy_analytic(np.linspace(0, 2, nx))

plt.plot(x_space, y_space, "blue")



psy_fd = np.zeros_like(y_space)
psy_fd[0] = 0. # IC

for i in range(1, len(x_space)):
    psy_fd[i] = psy_fd[i-1] + B(x_space[i]) * dx - psy_fd[i-1] * A(x_space[i]) * dx
    
plt.plot(x_space, psy_fd, "orange")
plt.show()


# In[7]:


suma1 = 0.
suma2 = 0.
for i in range(nx):
    suma1 += (y_space[i] - psy_fd[i])**2
    suma2 += (y_space[i] - Psi_NN[i])**2
print("Error en diferencias finitas: ", np.sqrt(suma1))
print("Error con red neuronal: ", np.sqrt(suma2))


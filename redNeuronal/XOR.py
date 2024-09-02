from Dense import Dense
from Tanh import Tanh
from Losses import mse, msePrime
import numpy as np

# ([Datos], Arreglos, Cantidad Elementos por Arreglo, Agrupar el arreglo en 1 solo)

X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1)) 

network = [
    Dense(2, 3),
    Tanh(),
    Dense(3, 1),
    Tanh()
]

epochs = 1000 # Generaciones Entrenamiento
learningRate = 0.1 # Incrementar el factor de aprendizaje formenta neuronas muertas.

for i in range(epochs):

    error = 0

    for x, y in zip(X,Y): # Ingresar los datos a evaluar

        output = x # Ingresar resultados para calcular el output,

        for layer in network:
            output = layer.forward(output) # Realizar la multiplicacion de matrices, 

        error = mse(y, output) # Calcular el error del resultado brindado de la red correspondiente al dato ingresado en el foward propagation.

        #Proceso de aprendizaje, actualización de los parametros de entrada.

        # Es calculado la primera derivada del backtracing de la penultima neurona.
        grad = msePrime(y, output)

        for layer in reversed(network): # Iteramos el array de manera inversa.

            grad = layer.backward(grad, learningRate)

        error /= len(X)
        print('%d/%d, error = %f' % (i + 1, epochs, error))
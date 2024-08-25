from Layer import Layer
import numpy as np

class Dense(Layer): # Esta clase definira el método de foward progration.

    # En una red Dense Layer todas las capas de los nodos previos se encuentran interconectados, cada camino representa y tiene asociado un peso.
    # El resultado brindado a la siguiente capa es la sumatoria de todas las neuronas multiplicado por su camino o peso con la sumatoria de un "Bias".
    # Un Bias es un parametro entrenable.

    # Por la definición previa brindada sobre una Dense Layer todos los nodos son multiplicados por todos sus caminos, por ende es pausible simplificar la cantidad de computo realizando una multiplicación de matrices.
    # Para la multiplicación de matrices deben coincidir el número de columnas y fila tomando como referencia la primera matriz los pesos y la segunda las neuronas.
    # Es importante denotar que la multiplicación matricial no es una operación conmutativa (A*B != B*A), dando como resultado una matriz del tamaño de la ij.
    # i = Columnas Primer Operando, j = Filas Segundo Operando.

    def __init__(self, inputSize, outputSize):

        # Todos los pesos de cada camino son de manera aleatoria con el entrenamiento se reajustaran.
        # Generame un array outputSize Filas e inputSize Columnas (Interconectar todos las las neuronas entre capas)
    
        self.weights = np.random.randn(outputSize, inputSize)
        self.bias = np.random.rand(outputSize, 1)

    def foward(self, input):

        self.input = input

        # Realizame el producto punto o multiplicacion matricial de la matriz de pesos con el valor de cada neurona, al final sumale la función Bias.

        return np.dot(self.weights, self.input) + self.bias 
    
    def backward(self, outputGradient, learningRate):

        pass
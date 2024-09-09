import numpy as np
from Layer import Layer

class Reshape(Layer):

    # La red neuronal da como resultado en cada evaluación una matriz de N * N dimesiones.
    # Cuando requiere como output disponer de un vector columna.

    def __init__(self, inputShape, outputShape):

        self.inputShape = inputShape
        self.outputShape = outputShape

    def forward(self, input):

        return np.reshape(input, self.outputShape)
    
    def backward(self, outputGradient, learningRate):

        return np.reshape(outputGradient, self.inputShape)
from Layer import Layer
import numpy as np

class Activation(Layer):

    def __init__(self, activation, activationPrime):

        self.activation = activation
        self.activationPrime = activationPrime

    def forward(self, input):

        # Transladamos el factor de activación a la siguiente capa

        self.input = input
        return self.activation(self.input)    
    
    def backward(self, outputGradient, learningRate):

        # Necesita calcular la derivada respecto a su input para realizar el backward propagation.
        # Solamente es necesario calcular la derivada respecto a la función correspondiente por tal motivo acontece un Hadamard product en lugar de una multiplicación matricial.

        return np.multiply(outputGradient, self.activationPrime(self.input))
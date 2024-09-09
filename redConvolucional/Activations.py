import numpy as np  
from Activation import Activation

# La función de activicacion lógistica o sigmoidal es empleada para hacer clasificaciones binarias.
# Al solamente evaluar dos posibles estados es usada, adicionalmente no recibe valores negativos y el valor retornado oscila entre 0 a 1 

class Sigmoid(Activation):

    def __init__(self):

        def sigmoid(x):
            return 1/ (1 + np.exp(-x))
        
        def sigmoidPrime(x):

            s = sigmoid(x)
            return s * (1 - s)

            super().__init__(sigmoid, sigmoidPrime)
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

class Tanh(Activation):

    # Es empleada la función de tangente inversa como función de activación.
    # Es una función no lineal que su derivada comprende un rango de 0 a 1.

    def __init__(self):

        # Una funcion lambda es empleado para definir una función pequeña anonima, función sin nombre.

        tahn = lambda x: np.tanh(x)
        tahnPrime = lambda x: 1 - np.tanh(x) ** 2 # Derivada
        super().__init__(tahn, tahnPrime) # Llama el constuctor de la clase padre            
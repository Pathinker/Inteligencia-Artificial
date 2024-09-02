from Activation import Activation
import numpy as np

class Tanh(Activation):

    # Es empleada la función de tangente inversa como función de activación.
    # Es una función no lineal que su derivada comprende un rango de 0 a 1.

    def __init__(self):

        # Una funcion lambda es empleado para definir una función pequeña anonima, función sin nombre.

        tahn = lambda x: np.tanh(x)
        tahnPrime = lambda x: 1 - np.tanh(x) ** 2 # Derivada
        super().__init__(tahn, tahnPrime) # Llama el constuctor de la clase padre
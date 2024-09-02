import numpy as np

    # La función calculadora de perdidas es una elección arbitraria, acontece en el primer paso del backward propagation.
    # Uno de los pasos es hacer la derivada del input, por ende tambien es incorporado un método que lo compute.
    # Compara directamente el resultado verdadero con el esperado.

def mse(yTrue, yPred):

    # Promedio de los errores al cuadrado, el error cuadrático medio es empleado en las regresione simples para calcular el error. 

    return np.mean(np.power(yTrue - yPred, 2)) # Foruma general de error cuadrático medio

def msePrime(yTrue, yPred):

    return 2 * (yPred - yTrue) / np.size(yTrue)
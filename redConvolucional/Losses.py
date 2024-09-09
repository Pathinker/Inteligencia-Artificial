import numpy as np

# Función empleada para calcular el error (Comparar la respuesta con el output esperado)
# Bianry Cross Entropy o Entropia Cruzada evalua el vector columna de respuesta que contendra una respuesta comprendida entre 0 a 1.
# Mide el error referente a una escala binaria.

def binaryCrossEntropy(yTrue, yPred):

       return np.mean(-yTrue * np.log(yPred) - (1 - yTrue) * np.log(1 - yPred))

def binaryCrossEntropyPrime(yTrue, yPred):
       
       return ((1 - yTrue) / (1 - yPred) - yTrue / yPred) / np.size(yTrue)
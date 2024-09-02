from Layer import Layer
import numpy as np

class Dense(Layer): # Esta clase definira el método de forward progration.

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

    def forward(self, input):

        self.input = input

        # Realizame el producto punto o multiplicacion matricial de la matriz de pesos con el valor de cada neurona, al final sumale la función Bias.

        return np.dot(self.weights, self.input) + self.bias 
    
    def backward(self, outputGradient, learningRate):

        # outputGradient corresponde a la derivada o crecimiento del error respecto al resultado.
    
        # En el backward propagation ocurren múltiples derivadas, es necesario calcular la derivada del error especto a los parametros y el bias.
        # Permitiendo de esta manera actualizar sus valores al ser los parametros entrenables.
        # Posteriormente es calculada la derivada del input que sera pasada a las neuronas de capas previas.
        # Su derivada al igual que todas es expresada y desarollada utilizando la regla de la cadena, la cual es empleada para derivar composición funciones.
        # Una función compuesta, es una función evaluada en otra función.

        # Primeramente en el backward propagation evalua la derivada del error respecto al output, siendo descompuesto en las sigueintes dos derivadas:
        # Error respecto al W o set de parametros, y la derivada del error respecto a su bias. 
        # El error respecto W puede ser sustituido por el el error evaluado en w{ij}, al ejecutar su deriva obtenemos el  resultado del coeficiente del peso que corresponde a la neurona.
        # La derivada toma en consideración todos los outpus del peso asociado al no estar presentes su derivada es 0, ligando a una formula.
        # La formula al ser puesta en una matriz es posible identificar elementos repetidos desenbocando en un producto de matrices.
        # T hace alusión a la transpuerta de una matriz que consiste en cambiar filas por columnas, dicha matriz permite generar varias matrices efectuando operaciones con la misma, como es la simetrica y asimetrica.
        # Es realizado para cumplir los requisitos de la multiplicación matricial.

        weightGradient = np.dot(outputGradient, self.input.T)

        # Es calculado el derivada respecto al input, donde la respuesta estar conformada por todos mis outputs y repetirse en los demas resultados al ser una red densa o con todos su nodos conectados.
        # Brinda como resultado sus pesos, los cuales es susceptible de efectuar una multiplicación matricial respecto las derivas del error respecto al output.
        
        inputGradient = np.dot(self.weights.T, outputGradient)

        # Actualización de los parametrosm, es multiplicado por un algoritmo optimizador deniminado learningRate.

        self.weights -= learningRate * weightGradient
        self.bias -= learningRate * outputGradient

        # Es importante retorna la derivada respecto al input al subsecuente layer o capa.

        return inputGradient
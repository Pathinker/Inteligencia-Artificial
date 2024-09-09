import numpy as np
from Layer import Layer
from scipy import signal

# Una red convolucional emplea la correlación cruzada "cross correlation" de mis matrices de inputs con un kernel.
# El kernel es suceptible de disponer de valores arbitrarios que sean útiles para la abstracción de caracteristicas.
# En otras palabras filtran los datos de entrada, considerando de igual manera el bias o el sezgo.

class Convolutional(Layer):

    def __init__(self, inputShape, kernelSize, depth):

        # Obtener la cantidad de matrices de entrada y sus dimenciones respectivamente, realiza un desempaquetado de los datos.
        inputDepth, inputHeight, inputWidth = inputShape
        self.depth = depth
        self.inputShape = inputShape
        self.inputDepth = inputDepth
        # Número de kernels y las dimenciones de las matrices.
        self.outputShape = (depth, inputHeight - kernelSize + 1, inputWidth - kernelSize + 1)
        # Numero de matrices de la capa de entrada, ya que a cada matriz le corresponde un kernel de longitud cuadrada.
        self.kernelsShape = (depth, inputDepth, kernelSize, kernelSize)
        # Inicializar de manera aleatoria.
        self.kernels = np.random.rand(*self.kernelsShape)
        self.biases = np.random.rand(*self.outputShape)

    def forward(self, input):

        # El forward propagation de una red neuronal es una suma ponderada de todos sus neuronas por sus pesos más un sezgo "bias".
        # En una red neuronal correlacional es repetida con la diferencia de ser matrices en lugar de productos escalares, por ende no es posible decomponerlo en productos matriciales.
        # Existen diferentes maneras de realizar correlaciones cruzadas.
        # "Valid" genera una matriz del tamaño del kernel donde solamente es evaluado el kernel respecto al input sin exederse de las longitudes de la matriz.
        # "Full" calcula todas las correlacones validas y es tomado en consideración los bordes donde valid no evalua, es obtenida una matriz de una mayor dimensión.

        self.input = input

        # En la correlación cruzada siempre es adicionado el sezgo o bias, por ende es copiado.

        self.output = np.copy(self.biases)

        for i in range(self.depth):
            for j in range(self.depth):

                # Empleamos la libreria spicy para ejecturar la cross correlation valid, es importante denotar que es no es conmutativo.
                # Al disponer del sezgo es sumdo al resultado de la corss correlation entre los inputs y los kernels.
                # Nos dara como resultado una matriz del mismo tamaño que el kernel por ende lo ejecutamos n veces siendo la profunidad y n veces siendo la cantidad de datos.

                self.output[i] += signal.correlate2d(self.input[j], self.kernels[i, j], "valid")

        return self.output
    
    def backward(self, outputGradient, learningRate):

        # Método de ajuste y aprendizaje sobre la comparación del resultado valido al esperado.

        # Inicializar en 0 las matrices empleadas para retonar el desenso de gradiente.

        kernelsGradient = np.zeros(self.kernelsShape)
        inputGradient = np.zeros(self.inputShape)

        # Ejecutamos el calculo del desenso de gradiente mediante las derivadas despejadas.
        # dE/dX = Convolucion completa del kernel con el output.
        # dE/Dk = Correlación cruzada del kernel con el input.
        # dE/dB = Es directamente el parametro "outputGradient" ya que represente la derivada del error con respecto del resultado.

        for i in range(self.depth):
            for j in range(self.depth):

                kernelsGradient[i][j] = signal.correlate2d(self.input[j], outputGradient[i], "valid") # dE/dK
                inputGradient[i][j] = signal.convolve2d(outputGradient[i], self.kernels[i][j], "full") # dE/dX


        self.kernels -= learningRate * kernelsGradient
        self.biases -= learningRate * outputGradient

        return inputGradient # Retornar en backpropagation.
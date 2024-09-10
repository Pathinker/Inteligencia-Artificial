import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical

from Dense import Dense
from Convolutional import Convolutional
from Reshape import Reshape
from Activations import Tanh, Sigmoid
from Losses import binaryCrossEntropy, binaryCrossEntropyPrime

def preprocessData(x, y, limit):

    # Indices de las dos objetos a identificar.

    zeroIndex = np.where(y == 0)[0][:limit]
    oneIndex = np.where(y == 1)[0][:limit]

    # Juntar los indices en un solo array y mezclarlos aleatoriamente.

    allIndices = np.hstack((zeroIndex, oneIndex))
    allIndices = np.random.permutation(allIndices)

    # Extraer todos las imagenes apartir de los indices.

    x, y = x[allIndices], y[allIndices]

    # Modificar el dataset de 28 x 28 pixels agregandole una dimensión, ya que el primer parametro que evalua la red es la profundidad.
    # Profundidad = Cantidad de elementos de entrada.

    x = x.reshape(len(x), 1, 28, 28)

    # Las imagenes tienen una profundidad de color de 8 bits, por ende es normalizado para disponer unicamente 1 bit de profunidad de color.
    # Solamente blanco o negro.

    x = x.astype("float32") /255

    # Asocia un identificador a cada uno de los datos, entrenamiento supervisado, posteriormente es almacenado en un vector de 2 columnas de una sola dimensión.

    y = to_categorical(y)
    y = y.reshape(len(y), 2, 1)

    return x, y

# Cargar los datos y procesarlos con 100 imagenes.

(xTrain, yTrain), (xTest, yTest) = mnist.load_data()
xTrain, yTrain = preprocessData(xTrain, yTrain, 100)
xTest, yTest = preprocessData(xTest, yTest, 100)

network = [

    # Indicamos la presencia de 5 kernels de 3x3 de tamaño y matrices de entrada de 28x28 con 1 de profundidad.
    # 28x28 representa el número total de pixeles de la imagen a capturar que son mis elementos de entrada.

    Convolutional((1, 28, 28), 3, 5),

    # Función de activación logistica brinda un resultado binario.

    Sigmoid(),

    # El primer array son las dimenciones de entrada.
    # El segundo las de salida que son emitidas en un vector columna, convertir la respuesta matriciales en un vector columna.

    Reshape((5, 26, 26), (5 * 26 * 26, 1)),

    # Indicamos la cantidad de neuronas que son exactamente la cantidad que la capa previa le provee, adicionalmente emite 100 respuestas.

    Dense(5 * 26 * 26, 100),

    Sigmoid(),

    # Obtenemos las 100 outputs de la capa previa y creamos solamente 2 salidas

    Dense(100, 2),

    Sigmoid()
]

epochs = 20
learningRate = 0.1

# Entrenamiento

for e in range(epochs):

    error = 0

    for x, y in zip(xTrain, yTrain):

        output = x

        # Obtener el elemento de entrada y realizar la forward propagation.

        for layer in network:

            output = layer.forward(output)

        error += binaryCrossEntropy(y, output)

        # Backpropagation

        grad = binaryCrossEntropyPrime(y, output)

        for layer in reversed(network):
            grad = layer.backward(grad, learningRate)

    error /= len(xTrain)
    print("{} / {}, Error =  {}".format(e + 1, epochs, error))

# Testear al hacer predicciones.

for x, y in zip(xTest, yTest):

    output = x

    for layer in network:
        output = layer.forward(output)

    print(f"Prediccion: {np.argmax(output)}, Resultado Esperado: {np.argmax(y)}")
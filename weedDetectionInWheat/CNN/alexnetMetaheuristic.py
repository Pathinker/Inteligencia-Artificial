import sys
import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Añadir la carpeta necesaria para importaqr la clase alexnet

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from weedDetectionInWheat.CNN.alexnetClass import alexnet
#from weedDetectionInWheat.metaheuristic.ADSCFGWO import ADSCFGWO
#from weedDetectionInWheat.metaheuristic.ADSCFGWOclassWeight import ADSCFGWO
#from weedDetectionInWheat.metaheuristic.GWO2 import GWO
from weedDetectionInWheat.metaheuristic.GWOGPU import GWO

# Cargar el set de datos
direccionDataset = Path("weedDetectionInWheat/Dataset")
direccionEntrenamiento = direccionDataset / "train/"
direccionValidamiento = direccionDataset / "valid/"

# Especificar las dimensiones de las imágenes y el tamaño de lotes
anchoImagen = 227
largoImagen = 227
imgSize = [anchoImagen, largoImagen]
batchSize = 32 

# Crear los dataframes de entrenamiento y validación
trainDataFrame = tf.keras.utils.image_dataset_from_directory(
    direccionEntrenamiento,
    seed=123,
    image_size=imgSize,
    batch_size=batchSize,
    label_mode="binary"
)

validacionDataFrame = tf.keras.utils.image_dataset_from_directory(
    direccionValidamiento,
    seed=123,
    image_size=imgSize,
    batch_size=batchSize,
    label_mode="binary"
)

dataArgumentation = tf.keras.Sequential([

    # Transformaciones Geometricas

    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomTranslation(0.1, 0.1),

    # Transformaciones Color       

    tf.keras.layers.RandomContrast(0.2),           # Ajuste del contraste
    tf.keras.layers.RandomBrightness(0.2),         # Ajuste del brillo                               
                                          
])

def procesarImagen(x, y):

    return dataArgumentation(x), y

# Modificar el dataset.

dataArgumentationTrain = trainDataFrame.map(procesarImagen)

# Construir el modelo AlexNet
CNN = alexnet(trainDataFrame, validacionDataFrame)
arquitecturaCNN = CNN.obtenerModelo()
pesosClases = CNN.obtenerPesosClases()

# Compilar y entrenar el modelo
arquitecturaCNN.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.Adam(0.001),
    metrics=['accuracy']
)

# Inicializar GWO con la estructura de los pesos del modelo
gwo = GWO(model=arquitecturaCNN, iterMaximo=20, classWeight=pesosClases)

# Optimizar con GWO
best_weights = gwo.optimize(dataArgumentationTrain, validacionDataFrame)

# Establecer los mejores pesos encontrados al modelo

#arquitecturaCNN.set_weights(best_weights)
arquitecturaCNN.save('weedDetectionInWheat/CNN/alexnetMetaheuristic.keras')
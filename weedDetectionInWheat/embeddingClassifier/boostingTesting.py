import sys
import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Input # type: ignore
from sklearn.svm import SVC # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
from sklearn.pipeline import Pipeline # type: ignore
from sklearn.metrics import accuracy_score # type: ignore

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from weedDetectionInWheat.metaheuristic.customLayers.maskLayer import MaskLayer

# Cargar datos predicción

anchoImagen = 227
largoImagen = 227
canales = 3 
batchSize = 32
imgSize = [anchoImagen, largoImagen]

# Cargar el set de datos CNN tensorflow.

direccionDataset = Path("weedDetectionInWheat/Dataset")
direccionEntrenamiento = direccionDataset / "train/"
direccionValidamiento = direccionDataset / "valid/"

trainDataFrame = tf.keras.utils.image_dataset_from_directory(

    direccionEntrenamiento,
    seed = 123,
    image_size = imgSize,
    batch_size = batchSize,
    label_mode = "binary"

)

validacionDataFrame = tf.keras.utils.image_dataset_from_directory(

    direccionValidamiento,
    seed=123,
    image_size=imgSize,
    batch_size=batchSize,
    label_mode="binary",
    #shuffle=False 

)

# Cargar el modelo al volver a cargarse no tiene un tipo de entrada definida por ende es computado una sola vez debido a ser un modelo secuencial.

alexnet = keras.models.load_model("weedDetectionInWheat/CNN/alexnetMetaheuristic.keras", custom_objects={'MaskLayer': MaskLayer})

alexnet.evaluate(validacionDataFrame, verbose = 1)
alexnet.summary()

# Extraer hasta la capa flatten deseada para el entrenamiento del SVM.

nombreCapa = "conv2d"
capaInicial = alexnet.get_layer(nombreCapa)

nombreCapa = "mask"  
capaObjetivo = alexnet.get_layer(nombreCapa)

# Crear nuevo modelo hasta la capa flatten siendo la última convolucional.

alexnetFlatten = Model(inputs = capaInicial.input, outputs = capaObjetivo.output)
alexnetFlatten.summary()

for images, etiquetas in trainDataFrame.take(1):

    imagen = images[0]
    imagen = tf.expand_dims(imagen, axis = 0)
    output = alexnetFlatten.predict(imagen)
    output = output.flatten()
    
    frecuenciaValores = {valor: (output == valor).sum() for valor in set(output)}
    frecuenciaOrdenada = dict(sorted(frecuenciaValores.items(), key=lambda item: item[1], reverse=True))
    print(frecuenciaOrdenada)

    print("Pesos: \n", capaObjetivo.get_weights())
import pickle
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from tensorflow import keras
from pathlib import Path
from sklearn.svm import SVC # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
from sklearn.pipeline import Pipeline # type: ignore

import sys
import os

# Añadir la carpeta necesaria para importaqr la clase alexnet

sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/../..')) 

from weedDetectionInWheat.CNN.alexnetClass import alexnet

# Bagging necesita crear subsets de entrenamiento y luego someter los modelos a un voting classifier.

# Cargar el set de datos.

direccionDataset = Path("weedDetectionInWheat/Dataset")
direccionEntrenamiento = direccionDataset / "train/"
direccionValidamiento = direccionDataset / "valid/"

# Especificar las dimensiones de las imagenes y el tamaño de lotes.

anchoImagen = 227
largoImagen = 227
canales = 3 
imgSize = [anchoImagen, largoImagen]
batchSize = 32 

# Crear los dataframes.

trainDataFrame = tf.keras.utils.image_dataset_from_directory(

    direccionEntrenamiento,
    seed = 123,
    image_size = imgSize,
    batch_size = batchSize,
    label_mode = "binary"

)

validacionDataFrame = tf.keras.utils.image_dataset_from_directory(

    direccionValidamiento,
    seed = 123,
    image_size = imgSize,
    batch_size = batchSize,
    label_mode = "binary"

)

# Numero total de imagenes

imagenesTotales = tf.data.experimental.cardinality(trainDataFrame).numpy()

# Definir el tamaño del lote como la mitad del total de imágenes

subset = imagenesTotales // 2

# Crear dos subconjuntos

primerLote = trainDataFrame.take(subset)  # Primer lote
segundoLote = trainDataFrame.skip(subset)  # Segundo lote

#primerLote = primerLote.batch(batchSize)

CNN = alexnet(primerLote, validacionDataFrame, "alexnetBagging.keras")

# -----------------------------------#

#  Entrenamiento SVM Radial

# -----------------------------------#

elementosEntrenamiento = []
etiquetasEntrenamiento = []

for featureBatch, labelBatch in segundoLote:
    
    # Iterar sobre cada imagen en el lote
    for feature, label in zip(featureBatch, labelBatch):

        elementosEntrenamiento.append(feature.numpy().flatten())
        etiquetasEntrenamiento.append(label.numpy())

elementosEvaluar = []
etiquetasEvaluar = []

for featureBatch, labelBatch in validacionDataFrame:
    
    for feature, label in zip(featureBatch, labelBatch):

        elementosEvaluar.append(feature.numpy().flatten())
        etiquetasEvaluar.append(label.numpy())

# Son asignados los labels de entrenamineto y test.

xTrain = np.array(elementosEntrenamiento)
yTrain = np.array(etiquetasEntrenamiento).ravel() # Vector 1D
xTest = np.array(elementosEvaluar)
yTest = np.array(etiquetasEvaluar).ravel() # Vector 1D

# Es entrenado un modelo con un radial kernel que interactua como weight nearest neighbor 

model = Pipeline([

    ("scaler", StandardScaler()),
    ("svm", SVC(C = 1, kernel = "rbf", gamma = "scale", verbose = True))

])

model.fit(xTrain, yTrain)

# Guardar Modelo

pick = open("weedDetectionInWheat/SVM/SVMrbfBagging.sav", "wb")
pickle.dump(model, pick)
pick.close()

# Cargar Modelo

pick = open("weedDetectionInWheat/SVM/SVMrbfBagging.sav", "rb")
model = pickle.load(pick)
pick.close()

prediciones = model.predict(xTest)
accurancy = model.score(xTest, yTest)

print("Accuracy Radial: ", accurancy)
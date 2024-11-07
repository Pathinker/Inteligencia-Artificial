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
    shuffle=False 

)

# Cargar el modelo al volver a cargarse no tiene un tipo de entrada definida por ende es computado una sola vez debido a ser un modelo secuencial.

alexnet = keras.models.load_model("weedDetectionInWheat/CNN/alexnet.keras")

alexnet.evaluate(validacionDataFrame, verbose = 1)

# Extraer hasta la capa flatten deseada para el entrenamiento del SVM.

nombreCapa = "conv2d"
capaInicial = alexnet.get_layer(nombreCapa)

nombreCapa = "flatten"  
capaObjetivo = alexnet.get_layer(nombreCapa)

# Crear nuevo modelo hasta la capa flatten siendo la última convolucional.

alexnetFlatten = Model(inputs = capaInicial.input, outputs = capaObjetivo.output)
alexnetFlatten.summary()

# Extraer las caracteristicas haciendo uso de la capas convolucionales del modelo alexnet

def extraerConvolucion(dataset):

    features = []
    labels = []
    
    for images, batch_labels in dataset:

        # Extraer características de la capa flatten de cada una de las imágenes
        batch_features = alexnetFlatten(images, training=False)
        features.append(batch_features.numpy())  # Convertir a numpy
        labels.append(batch_labels.numpy())  # Obtener las etiquetas
    
    return np.concatenate(features), np.concatenate(labels)

model = Pipeline([

    ("scaler", StandardScaler()),
    ("svm", SVC(C = 1, kernel = "rbf", gamma = "scale", verbose = True))

])

# Extraer características y etiquetas
xTrain, yTrain = extraerConvolucion(trainDataFrame)
yTrain = yTrain.ravel() # Transformar a 1D

# Entrenar el SVM con las características extraídas
model.fit(xTrain, yTrain)

pick = open("weedDetectionInWheat/SVM/SVMrbfBoosting.sav", "wb")
pickle.dump(model, pick)
pick.close()

# Validar el modelo

xValidacion, yValidacion = extraerConvolucion(validacionDataFrame)
yValidacion = yValidacion.ravel() # Transformar a 1D
yPrediccion = model.predict(xValidacion)

# Calcular precisión
accuracy = accuracy_score(yValidacion, yPrediccion)
print(f"Precisión del modelo SVM Boosting CNN: {accuracy}")
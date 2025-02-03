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
from scipy.special import expit # type: ignore

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
)

# Cargar el modelo al volver a cargarse no tiene un tipo de entrada definida por ende es computado una sola vez debido a ser un modelo secuencial.

alexnet = keras.models.load_model("weedDetectionInWheat/CNN/alexnet_combined.keras")

alexnet.evaluate(validacionDataFrame, verbose = 1)
alexnet.summary()

# Extraer hasta la capa flatten deseada para el entrenamiento del SVM.

nombreCapa = "mask"  
capaObjetivo = alexnet.get_layer(nombreCapa)

# Crear nuevo modelo hasta la capa flatten siendo la última convolucional.

alexnetFlatten = Model(inputs = alexnet.input, outputs = capaObjetivo.output)
alexnetFlatten.summary()

pickIn = open("weedDetectionInWheat/SVM/SVMrbfBoostingMetaheuristic.sav", "rb")
SVM = pickle.load(pickIn)
pickIn.close()

def soft_voting(dataset):
    features = []
    labels = []
    alexnet_preds = []

    for images, batch_labels in dataset:
        batch_features = alexnetFlatten(images, training=False).numpy()
        features.append(batch_features)
        labels.append(batch_labels.numpy())

        batch_preds = alexnet(images, training=False).numpy().flatten()
        alexnet_preds.append(batch_preds)

    x = np.concatenate(features, axis=0)
    y = np.concatenate(labels, axis=0)
    yPredAlexNet = np.concatenate(alexnet_preds, axis=0)
    ySVM = expit(SVM.decision_function(x))

    yFinal = (yPredAlexNet + ySVM) / 2
    yFinal = (yFinal > 0.5).astype(int) 

    return accuracy_score(y, yFinal)

print(f"Precisión del modelo SVM Boosting Soft Voting Train: {soft_voting(trainDataFrame)}")
print(f"Precisión del modelo SVM Boosting Soft Voting Validation: {soft_voting(validacionDataFrame)}")
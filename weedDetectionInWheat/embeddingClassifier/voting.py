import pickle
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from tensorflow import keras
from pathlib import Path
from tensorflow.keras.models import load_model # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore

# Cargar el modelo convolucional

alexnet = load_model("weedDetectionInWheat/CNN/alexnet.keras")

# Cargar SVM Kernel Radial (Mejor Performance)

pickIn = open("weedDetectionInWheat/SVM/SVMrbf.sav", "rb")
SVM = pickle.load(pickIn)
pickIn.close()

# Cargar datos predicción

anchoImagen = 227
largoImagen = 227
canales = 3 
batchSize = 32
imgSize = [anchoImagen, largoImagen]

# Cargar el set de datos CNN tensorflow.

direccionDataset = Path("weedDetectionInWheat/Dataset")
direccionValidamiento = direccionDataset / "valid/"

validacionDataFrame = tf.keras.utils.image_dataset_from_directory(

    direccionValidamiento,
    seed=123,
    image_size=imgSize,
    batch_size=batchSize,
    label_mode="binary",
    shuffle=False 

)

# Aplanar y normalizar imágenes para el SVM

imagenesValidacion = []
etiquetasValidacion = []

for batch in validacionDataFrame:
    
    images, labels = batch  # Separar imágenes y etiquetas
    images_reshaped = tf.reshape(images, (images.shape[0], anchoImagen * largoImagen * canales))
    imagenesValidacion.append(images_reshaped.numpy())  # Convertir a numpy array y guardar
    etiquetasValidacion.append(labels.numpy())  # Guardar etiquetas

imagenesValidacion = np.concatenate(imagenesValidacion)
etiquetasValidacion = np.concatenate(etiquetasValidacion)

# Normalizar las imágenes para SVM
scaler = StandardScaler()
elementosEvaluarNormalizacion = scaler.fit_transform(imagenesValidacion)

# Predicción con el modelo CNN
prediccionesCNN = alexnet.predict(validacionDataFrame)  # Probabilidades de la CNN

# Predicción con el modelo SVM (probabilidades)
prediccionesSVM = SVM.predict_proba(elementosEvaluarNormalizacion)[:, 1]  # Probabilidad de clase 1

# Soft Voting (promedio de probabilidades)
prediccionesCombinadas = (prediccionesCNN.flatten() + prediccionesSVM) / 2  # Promedio de las probabilidades

# Clasificar basado en el promedio de las probabilidades (umbral 0.5)
prediccionesFinales = np.where(prediccionesCombinadas > 0.5, 1, 0)

# Evaluar la precisión comparando con las etiquetas reales
precision = np.mean(prediccionesFinales == etiquetasValidacion)
print(f"Precisión del Voting Classifier (Soft Voting): {precision:.2f}")
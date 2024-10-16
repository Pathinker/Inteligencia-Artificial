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

# Transformar el conjunto de datos a la forma (32, 154587)
imagenesValidacion = []

for batch in validacionDataFrame:

    images, labels = batch  # Separar imágenes y etiquetas
    # Aplanar cada imagen a una dimensión

    images_reshaped = tf.reshape(images, (images.shape[0], anchoImagen * largoImagen * canales))
    imagenesValidacion.append(images_reshaped.numpy())  # Convertir a numpy array y guardar

scaler = StandardScaler()
elementosEvaluarNormalizacion = scaler.fit_transform(imagenesValidacion)

# Predicción con el modelo CNN
prediccionesCNN = alexnet.predict(validacionDataFrame)
prediccionesCNNClasificadas = np.where(prediccionesCNN > 0.5, 1, 0)

# Predicción con el modelo SVM
prediccionesSVM = SVM.predict(elementosEvaluarNormalizacion)

# Unir las predicciones
prediccionFinal = []

for claseCNN, claseSVM in tqdm(zip(prediccionesCNNClasificadas, prediccionesSVM), total=len(prediccionesCNNClasificadas)):
    # Usar votación por mayoría
    if claseCNN == claseSVM:
        prediccionFinal.append(claseCNN)
    else:
        # Si hay un empate, puedes elegir una estrategia para romper el empate.
        # Aquí se escoge el SVM como desempate, pero puedes elegir otra estrategia.
        prediccionFinal.append(claseSVM)

prediccionesFinales = np.array(prediccionFinal)

precision = np.mean(prediccionesFinales == np.array(imagenesValidacion))
print(f"Precisión del Voting Classifier: {precision:.2f}")
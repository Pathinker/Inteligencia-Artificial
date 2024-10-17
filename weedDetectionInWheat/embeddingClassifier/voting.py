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

pickIn = open("weedDetectionInWheat/SVM/SVMrbfr.sav", "rb")
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
etiquetasValidacion = []

for batch in validacionDataFrame:

    images, labels = batch  # Separar imágenes y etiquetas
    # Aplanar cada imagen a una dimensión

    images_reshaped = tf.reshape(images, (images.shape[0], anchoImagen * largoImagen * canales))
    imagenesValidacion.append(images_reshaped.numpy())  # Convertir a numpy array y guardar
    etiquetasValidacion.append(labels.numpy())

imagenesValidacion = np.vstack(imagenesValidacion)
etiquetasValidacion = np.concatenate(etiquetasValidacion)

# Predicción con el modelo CNN (obtener probabilidades en lugar de clases)
probabilidadesCNN = alexnet.predict(validacionDataFrame)
#print(probabilidadesCNN)

# Predicción con el modelo SVM
probabilidadesSVM = SVM.decision_function(imagenesValidacion)
#print(probabilidadesSVM)

# Unir las predicciones
prediccionFinal = []

for probCNN, probSVM in tqdm(zip(probabilidadesCNN.flatten(), probabilidadesSVM.flatten()), total=len(probabilidadesCNN)):
    
    # Promediar las probabilidades
    soft_vote = (probCNN + probSVM) / 2
    
    # Convertir las probabilidades a una clase final (0 o 1)
    claseFinal = 1 if soft_vote > 0.5 else 0
    prediccionFinal.append(claseFinal)

prediccionesFinales = np.array(prediccionFinal)

precision = np.mean(prediccionesFinales == etiquetasValidacion.flatten())
print(f"Precisión del Voting Classifier: {precision}")
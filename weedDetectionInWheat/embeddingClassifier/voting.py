import pickle
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from tensorflow import keras
from pathlib import Path
from tensorflow.keras.models import load_model # type: ignore

def predicciones(SVM, Modelo):

    # Predicción con el modelo SVM
    probabilidadesSVM = SVM.decision_function(imagenesValidacion)

    # Unir las predicciones
    prediccionFinal = []

    for probCNN, probSVM in tqdm(zip(probabilidadesCNN.flatten(), probabilidadesSVM.flatten()), total=len(probabilidadesCNN)):
        
        # Promediar las probabilidades para el sofvoting

        softVoting = (probCNN + probSVM) / 2
        prediccion = 0
        
        # Convertir las probabilidades a una clase final (0 o 1)

        if(softVoting > 0.5):
            prediccion = 1

        prediccionFinal.append(prediccion)

    prediccionesFinales = np.array(prediccionFinal)

    precision = np.mean(prediccionesFinales == etiquetasValidacion.flatten())
    print(f"Precisión del Voting Classifier SVM {Modelo}: {precision}")


# Cargar el modelo convolucional

alexnet = load_model("weedDetectionInWheat/CNN/alexnet.keras")

# Cargar SVM Kernel Radial (Mejor Performance)

pickIn = open("weedDetectionInWheat/SVM/SVMlinear.sav", "rb")
SVMLineal = pickle.load(pickIn)
pickIn.close()

pickIn = open("weedDetectionInWheat/SVM/SVMpolynomial.sav", "rb")
SVMPolynomial = pickle.load(pickIn)
pickIn.close()

pickIn = open("weedDetectionInWheat/SVM/SVMrbf.sav", "rb")
SVMRadial = pickle.load(pickIn)
pickIn.close()

pickIn = open("weedDetectionInWheat/SVM/SVMVotingBatches.sav", "rb")
SVMRadialBatches = pickle.load(pickIn)
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

    # Separar el label y la imagen de cada batch

    images, labels = batch 

    # Transformar al formato de entrada del SVM

    imagenesReshaped = tf.reshape(images, (images.shape[0], anchoImagen * largoImagen * canales))
    imagenesValidacion.append(imagenesReshaped.numpy())
    etiquetasValidacion.append(labels.numpy())

# Agrupar las imagenes y etiquetas   

imagenesValidacion = np.vstack(imagenesValidacion)
etiquetasValidacion = np.concatenate(etiquetasValidacion)

# Predicción con el modelo CNN (obtener probabilidades en lugar de clases)
probabilidadesCNN = alexnet.predict(validacionDataFrame)

predicciones(SVMLineal, "Lineal")
predicciones(SVMPolynomial, "Polinomial")
predicciones(SVMRadial, "Radial")
predicciones(SVMRadialBatches, "Radial Batches")
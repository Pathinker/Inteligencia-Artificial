import os 
import numpy as np
import cv2
import pickle

from tqdm import tqdm
from pathlib import Path

direccionDataset = Path("weedDetectionInWheat/Dataset")
direccionEntrenamiento = direccionDataset / "train/"
direccionValidamiento = direccionDataset / "valid/"

# Necesito extraer las categorias de clasificación para el SVM.

categorias = []
dataEntrenamiento = []
dataEvaluar = []

for subfolder in direccionEntrenamiento.iterdir():

    if subfolder.is_dir():
        categorias.append(subfolder.name)

# Preporcesar todas las imagenes.

for folder in categorias:

    direccion = os.path.join(direccionEntrenamiento, folder)
    etiquetas = categorias.index(folder)

    # Incorporamos con rqdm una herramienta visual para contemplar el progreso del cargado de imagenes.
    # Para usarlo solamente es incorporado en el for como prefijo, consecuentemente la básica de algoritmo y consecuentemente el mensaje a mostar.

    for imagen in tqdm(os.listdir(direccion), desc=f"Procesando Entrenamiento: {folder}"):

        direccionImagen = os.path.join(direccion, imagen)
        imagenEntrenamiento = cv2.imread(direccionImagen, 3)
        imagenEntrenamiento = cv2.resize(imagenEntrenamiento,(227, 227))
        imagen = np.array(imagenEntrenamiento).flatten()

        dataEntrenamiento.append([imagen, etiquetas])

for folder in categorias:

    direccion = os.path.join(direccionValidamiento, folder)
    etiquetas = categorias.index(folder)

    # Incorporamos con rqdm una herramienta visual para contemplar el progreso del cargado de imagenes.
    # Para usarlo solamente es incorporado en el for como prefijo, consecuentemente la básica de algoritmo y consecuentemente el mensaje a mostar.

    for imagen in tqdm(os.listdir(direccion), desc=f"Procesando Evaluar: {folder}"):

        direccionImagen = os.path.join(direccion, imagen)
        imagenEntrenamiento = cv2.imread(direccionImagen, 3)
        imagenEntrenamiento = cv2.resize(imagenEntrenamiento,(227, 227))
        imagen = np.array(imagenEntrenamiento).flatten()

        dataEvaluar.append([imagen, etiquetas])

# Transformar en una archivo load para ser usado por el modelo SVM.

pickIn = open("weedDetectionInWheat/SVM/dataEntrenamiento.pickle", "wb")
pickle.dump(dataEntrenamiento, pickIn)
pickIn.close()

pickIn = open("weedDetectionInWheat/SVM/dataEvaluar.pickle", "wb")
pickle.dump(dataEvaluar, pickIn)
pickIn.close()
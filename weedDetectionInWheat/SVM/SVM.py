# --- El archivo requiere la ejecución previa preprocesamiento.py para generar el archivo picke de lectura de las imagenes ---

import os 
import random
import numpy as np
import cv2
import pickle

from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.svm import SVC # type: ignore

# Cargar imagenes desde el archivo picke

pickIn = open("weedDetectionInWheat/SVM/data.pickle", "rb")
imagenes = pickle.load(pickIn)
pickIn.close()

random.shuffle(imagenes)

elementos = []
etiquetas = []

for feature, label in imagenes:

    elementos.append(feature)
    etiquetas.append(label)

xTrain, xTest, yTrain, yTest = train_test_split(elementos, etiquetas, test_size = 0.10)

model = SVC(C = 1, kernel = "poly", gamma = "auto", verbose = True)
model.fit(xTrain, yTrain)

prediciones = model.predict(xTest)
acurrancy = model.score(xTest, yTest)

pick = open("weedDetectionInWheat/SVM/SVM.sav", "rb")
model = pickle.load(pick)
pick.close()

print("Acurrancy: ", acurrancy)
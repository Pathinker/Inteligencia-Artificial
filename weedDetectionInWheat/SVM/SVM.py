# --- El archivo requiere la ejecución previa preprocesamiento.py para generar el archivo picke de lectura de las imagenes ---

import random
import pickle

from sklearn.svm import SVC # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore

# Cargar imagenes entrenamiento desde el archivo pickle

pickIn = open("weedDetectionInWheat/SVM/dataEntrenamiento.pickle", "rb")
imagenesEntrenamiento = pickle.load(pickIn)
pickIn.close()

# Cargar imagenes evaluación desde archivo pickle

pickIn = open("weedDetectionInWheat/SVM/dataEvaluar.pickle", "rb")
imagenesEvaluar = pickle.load(pickIn)
pickIn.close()

# Indicar la etiqueta pertenenciente

random.shuffle(imagenesEntrenamiento)

elementosEntrenamiento = []
etiquetasEntrenamiento = []

for feature, label in imagenesEntrenamiento:

    elementosEntrenamiento.append(feature)
    etiquetasEntrenamiento.append(label)

elementosEvaluar = []
etiquetasEvaluar = []

for feature, label in imagenesEvaluar:
    
    elementosEvaluar.append(feature)
    etiquetasEvaluar.append(label)

# Son asignados los labels de entrenamineto y test.

xTrain = elementosEntrenamiento
yTrain = etiquetasEntrenamiento
xTest = elementosEvaluar
yTest = etiquetasEvaluar

# Normalizar los datos
scaler = StandardScaler()
xTrain = scaler.fit_transform(xTrain)  # Ajusta y transforma los datos de entrenamiento
xTest = scaler.transform(xTest)  # Solo transforma los datos de test

# Es entrenado un modelo con un radial kernel que interactua como weight nearest neighbor 

model = SVC(C = 1, kernel = "linear", gamma = "scale", verbose = True)
model.fit(xTrain, yTrain)

# Guardar Modelo

pick = open("weedDetectionInWheat/SVM/SVMlinear.sav", "wb")
pickle.dump(model, pick)
pick.close()

# Cargar Modelo

pick = open("weedDetectionInWheat/SVM/SVMlinear.sav", "rb")
model = pickle.load(pick)
pick.close()

prediciones = model.predict(xTest)
accurancy = model.score(xTest, yTest)

print("Accuracy Lineal: ", accurancy)

# Es entrenado un modelo con un kernel polynomial

model = SVC(C = 1, kernel = "poly", gamma = "scale", verbose = True)
model.fit(xTrain, yTrain)

# Guardar Modelo

pick = open("weedDetectionInWheat/SVM/SVMpolynomial.sav", "wb")
pickle.dump(model, pick)
pick.close()

# Cargar Modelo

pick = open("weedDetectionInWheat/SVM/SVMpolynomial.sav", "rb")
model = pickle.load(pick)
pick.close()

prediciones = model.predict(xTest)
accurancy = model.score(xTest, yTest)

print("Accuracy Polynomial: ", accurancy)

# Es entrenado un modelo con un radial kernel que interactua como weight nearest neighbor 

model = SVC(C = 1, kernel = "rbf", gamma = "scale", verbose = True)
model.fit(xTrain, yTrain)

# Guardar Modelo

pick = open("weedDetectionInWheat/SVM/SVMrbf.sav", "wb")
pickle.dump(model, pick)
pick.close()

# Cargar Modelo

pick = open("weedDetectionInWheat/SVM/SVMrbf.sav", "rb")
model = pickle.load(pick)
pick.close()

prediciones = model.predict(xTest)
accurancy = model.score(xTest, yTest)

print("Accuracy Radial: ", accurancy)
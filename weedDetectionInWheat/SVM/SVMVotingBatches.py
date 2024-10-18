# --- El archivo requiere la ejecución previa preprocesamiento.py para generar el archivo picke de lectura de las imagenes ---

import random
import pickle

from sklearn.svm import SVC # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
from sklearn.pipeline import Pipeline # type: ignore
from tqdm import tqdm 

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

batchSize = 64

modelSVM = Pipeline([

    ("scaler", StandardScaler()),
    ("svm", SVC(C = 1, kernel = "rbf", gamma = "scale", probability = True))

])

# Función para dividir los datos en batches
def batch_data(x, y, batch_size):

    for i in range(0, len(x), batch_size):

        yield x[i:i + batch_size], y[i:i + batch_size] # Retorna un valor y permite su siguiente ejecución

# Entrenamiento Datos Entrenamiento

# Entrenamiento por batches con barra de progreso

numeroLotes = len(xTrain) // batchSize 

if(numeroLotes % batchSize != 0):
    numeroLotes += 1

for batchX, batchY in tqdm(batch_data(xTrain, yTrain, batchSize), total = numeroLotes,  desc="Entrenando SVM"):

    modelSVM.fit(batchX, batchY)

# Guardar Modelo

pick = open("weedDetectionInWheat/SVM/SVMVotingBatches.sav", "wb")
pickle.dump(modelSVM, pick)
pick.close()

# Cargar Modelo

pick = open("weedDetectionInWheat/SVM/SVMVotingBatches.sav", "rb")
model = pickle.load(pick)
pick.close()

prediciones = model.predict(xTest)
accurancy = model.score(xTest, yTest)

print("Accuracy Radial: ", accurancy)
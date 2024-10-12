import pickle
import numpy as np
from sklearn.decomposition import PCA # type: ignore
from sklearn.svm import SVC  # type: ignore

# Graficar en 2D los margin creados por cada uno de los modelos.
# Para ello es necesario crear modelos entrenados con una cantidad inferior de dimensiones

pickIn = open("weedDetectionInWheat/SVM/dataEvaluar.pickle", "rb")
imagenesEvaluar = pickle.load(pickIn)
pickIn.close()

elementos = []
etiquetas = []

for feature, label in imagenesEvaluar:

    elementos.append(feature)
    etiquetas.append(label)

# No es susceptible de graficar 227 * 227 * 3 = 154,857 dimensiones del dataset, para ello sera descomupuesto en 2 Dimensiones.

xTest = np.array(elementos)
yTest = np.array(etiquetas)

xFlatten = xTest.reshape(len(xTest), -1)

# Reducir las dimensiones con PCA a 2 componentes
pca = PCA(n_components=2)
xPCA = pca.fit_transform(xFlatten)

modeloPCA = SVC(C = 1, kernel='linear', gamma = "scale")
modeloPCA.fit(xPCA, yTest)

pick = open("weedDetectionInWheat/SVM/PCA/2Dlinear.sav", "wb")
pickle.dump(modeloPCA, pick)
pick.close()

print("Entrenamiento Lineal Finalizado.")

modeloPCA = SVC(C = 1, kernel = "poly", gamma = "scale")
modeloPCA.fit(xPCA, yTest)

pick = open("weedDetectionInWheat/SVM/PCA/2Dpolynomial.sav", "wb")
pickle.dump(modeloPCA, pick)
pick.close()

print("Entrenamiento Polynomial Finalizado.")

modeloPCA = SVC(C = 1, kernel = "rbf", gamma = "scale")
modeloPCA.fit(xPCA, yTest)

pick = open("weedDetectionInWheat/SVM/PCA/2Drbf.sav", "wb")
pickle.dump(modeloPCA, pick)
pick.close()

print("Entrenamiento Radial Finalizado.")
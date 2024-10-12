import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA # type: ignore

def graficar(modelo, xPCA, string):

    # Crear una malla para las coordenadas con un paso mayor (por ejemplo, 1.0 en lugar de 0.02)
    x_min, x_max = xPCA[:, 0].min() - 1, xPCA[:, 0].max() + 1
    y_min, y_max = xPCA[:, 1].min() - 1, xPCA[:, 1].max() + 1

    # Crear una malla para las coordenadas con un paso más grande para reducir el uso de memoria
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 10.0),
                        np.arange(y_min, y_max, 10.0))

    # Predecir para cada punto en la malla
    Z = modelo.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Graficar la malla y las fronteras de decisión
    plt.contourf(xx, yy, Z, alpha=0.8)

    # Graficar los puntos de datos originales
    plt.scatter(xPCA[yTest == 0, 0], xPCA[yTest == 0, 1], c='purple', edgecolor='k', label='Clase 0 (Weed)', s=30)
    plt.scatter(xPCA[yTest == 1, 0], xPCA[yTest == 1, 1], c='yellow', edgecolor='k', label='Clase 1 (No Weed)', s=30)

    # Quitar números laterales.

    plt.xticks([])  
    plt.yticks([])

    # Ajustar los límites de los ejes para evitar franjas blancas
    plt.xlim([x_min, x_max])
    plt.ylim([y_min, y_max])

    plt.title('Margin SVM kernel {} reducción de dimensiones (PCA)'.format(string))
    plt.legend(loc='upper right')
    plt.show()

# Cargar los modelos 2D para obtener una representación grafica de los resultados obtenidos.

pickIn = open("weedDetectionInWheat/SVM/dataEvaluar.pickle", "rb")
imagenesEvaluar = pickle.load(pickIn)
pickIn.close()

pickIn = open("weedDetectionInWheat/SVM/PCA/2Dlinear.sav", "rb")
modeloLineal = pickle.load(pickIn)
pickIn.close()

pickIn = open("weedDetectionInWheat/SVM/PCA/2Dpolynomial.sav", "rb")
modeloPolinomial = pickle.load(pickIn)
pickIn.close()

pickIn = open("weedDetectionInWheat/SVM/PCA/2Drbf.sav", "rb")
modeloRadial = pickle.load(pickIn)
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

graficar(modeloLineal, xPCA, "lineal")
graficar(modeloPolinomial, xPCA, "polinomial")
graficar(modeloRadial, xPCA, "radial")
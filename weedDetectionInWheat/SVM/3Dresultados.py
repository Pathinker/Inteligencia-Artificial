import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA  # type: ignore
from mpl_toolkits.mplot3d import Axes3D  # Para graficar en 3D
from matplotlib.colors import ListedColormap

def graficar(modelo, xPCA, string):

    # Crear una malla para las coordenadas en 3D
    x_min, x_max = xPCA[:, 0].min() - 1, xPCA[:, 0].max() + 1
    y_min, y_max = xPCA[:, 1].min() - 1, xPCA[:, 1].max() + 1
    z_min, z_max = xPCA[:, 2].min() - 1, xPCA[:, 2].max() + 1

    # Crear una malla 3D
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 300.0),
                         np.arange(y_min, y_max, 300.0))
    # Predecir para cada punto en la malla 3D
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = modelo.predict(np.c_[grid_points, np.zeros(grid_points.shape[0])])  # Asumir z = 0 para la predicción
    Z = Z.reshape(xx.shape)

    # Graficar los puntos de datos originales
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    custom_cmap = ListedColormap(['purple', 'yellow'])

    ax.scatter(xPCA[yTest == 0, 0], xPCA[yTest == 0, 1], xPCA[yTest == 0, 2], 
               c='purple', edgecolor='k', label='Clase 0 (Weed)', s=30)
    ax.scatter(xPCA[yTest == 1, 0], xPCA[yTest == 1, 1], xPCA[yTest == 1, 2], 
               c='yellow', edgecolor='k', label='Clase 1 (No Weed)', s=30)

    # Graficar la superficie de decisión
    ax.plot_surface(xx, yy, Z, alpha=0.5, cmap=custom_cmap, edgecolor='none')
    
    # Ajustar los límites de los ejes
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_zlim([z_min, z_max])

    # Quitar números laterales
    ax.set_xticks([])  
    ax.set_yticks([])
    ax.set_zticks([])

    ax.set_title('Margin SVM kernel {} reducción de dimensiones (PCA)'.format(string))
    ax.legend(loc='upper right')
    plt.show()

# Cargar los modelos 3D para obtener una representación grafica de los resultados obtenidos.

pickIn = open("weedDetectionInWheat/SVM/dataEvaluar.pickle", "rb")
imagenesEvaluar = pickle.load(pickIn)
pickIn.close()

pickIn = open("weedDetectionInWheat/SVM/PCA/3Dlinear.sav", "rb")
modeloLineal = pickle.load(pickIn)
pickIn.close()

pickIn = open("weedDetectionInWheat/SVM/PCA/3Dpolynomial.sav", "rb")
modeloPolinomial = pickle.load(pickIn)
pickIn.close()

pickIn = open("weedDetectionInWheat/SVM/PCA/3Drbf.sav", "rb")
modeloRadial = pickle.load(pickIn)
pickIn.close()

elementos = []
etiquetas = []

for feature, label in imagenesEvaluar:

    elementos.append(feature)
    etiquetas.append(label)

# No es susceptible de graficar 227 * 227 * 3 = 154,857 dimensiones del dataset, para ello sera descomupuesto en 3 Dimensiones.

xTest = np.array(elementos)
yTest = np.array(etiquetas)

xFlatten = xTest.reshape(len(xTest), -1)

# Reducir las dimensiones con PCA a 3 componentes
pca = PCA(n_components=3)
xPCA = pca.fit_transform(xFlatten)

graficar(modeloLineal, xPCA, "lineal")
graficar(modeloPolinomial, xPCA, "polinomial")
graficar(modeloRadial, xPCA, "radial")
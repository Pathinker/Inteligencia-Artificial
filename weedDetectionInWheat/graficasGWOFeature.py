import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

datos = np.loadtxt("weedDetectionInWheat/CNN/MetaheuristicReport.txt", delimiter=",")

epochs = []

for i in range(len(datos[0])):

    epochs.append(i + 1)

plt.title("Precisión Modelo")
plt.plot(epochs, datos[0], label = "Alfa", color = "red")
plt.plot(epochs, datos[5], label = "Beta", color = "blue")
plt.plot(epochs, datos[10], label = "Delta", color = "green")
plt.xlabel("Epochs")
plt.ylabel("Porcentaje")
plt.legend()
plt.show()

plt.title("Feature Selection")
plt.plot(epochs, datos[4], label = "Alfa", color = "red")
plt.plot(epochs, datos[9], label = "Beta", color = "blue")
plt.plot(epochs, datos[14], label = "Delta", color = "green")
plt.xlabel("Epochs")
plt.ylabel("Numero Features")
plt.legend()
plt.show()

plt.title("Error Modelo")
plt.plot(epochs, datos[1], label = "Alfa", color = "red")
plt.plot(epochs, datos[6], label = "Beta", color = "blue")
plt.plot(epochs, datos[11], label = "Delta", color = "green")
plt.xlabel("Epochs")
plt.ylabel("Error")
plt.legend()
plt.show()
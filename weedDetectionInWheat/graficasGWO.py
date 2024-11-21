import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

datos = np.loadtxt("weedDetectionInWheat/CNN/MetaheuristicReport.txt", delimiter=",")

epochs = []

for i in range(len(datos[0])):

    epochs.append(i + 1)

plt.title("Precisión Modelo")
plt.plot(epochs, datos[0], label = "Alfa", color = "red")
plt.plot(epochs, datos[4], label = "Beta", color = "blue")
plt.plot(epochs, datos[8], label = "Delta", color = "green")
plt.xlabel("Epochs")
plt.ylabel("Porcentaje")
plt.legend()
plt.show()

plt.title("Error Modelo")
plt.plot(epochs, datos[1], label = "Alfa", color = "red")
plt.plot(epochs, datos[5], label = "Beta", color = "blue")
plt.plot(epochs, datos[9], label = "Delta", color = "green")
plt.xlabel("Epochs")
plt.ylabel("Error")
plt.legend()
plt.show()

model = tf.keras.models.load_model('weedDetectionInWheat/CNN/alexnetMetaheuristic.keras')

weights = model.get_weights()
flattenedWeights = np.concatenate([weight.flatten() for weight in weights])

meanValue = np.mean(flattenedWeights)
variance = np.var(flattenedWeights)
stdDeviation = np.std(flattenedWeights)
maxValue = np.max(flattenedWeights)
minValue = np.min(flattenedWeights)

print(f"Mean: {meanValue}")
print(f"Variance: {variance}")
print(f"Standard Deviation: {stdDeviation}")
print(f"Max Value: {maxValue}")
print(f"Min Value: {minValue}")

plt.figure(figsize=(10, 6))
plt.boxplot(flattenedWeights, vert=False)
plt.title("Box Plot of Weights")
plt.xlabel("Weight Value")
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(flattenedWeights, bins=100, color='skyblue', edgecolor='black')
plt.title("Weights Histogram")
plt.xlabel("Weight Value")
plt.ylabel("Frequency")
plt.show()
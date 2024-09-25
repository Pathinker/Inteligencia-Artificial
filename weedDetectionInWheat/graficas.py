import matplotlib.pyplot as plt

primerAciertoEntrenamiento = [0.7070, 0.7724, 0.8059, 0.8272, 0.8319, 0.8450, 0.8465, 0.8658, 0.8590, 0.8621, 0.8685, 0.8790, 0.8918, 0.8871, 0.8872, 0.8910, 0.9037, 0.9071, 0.9108, 0.9168, 0.9140, 0.9293, 0.9308, 0.9363, 0.9313, 0.9426, 0.9433, 0.9489, 0.9542, 0.9570, 0.9617, 0.9639, 0.9603, 0.9732, 0.9697, 0.9715, 0.9760, 0.9779, 0.9745, 0.9823, 0.9807, 0.9852, 0.9896, 0.9870, 0.9881, 0.9869, 0.9858, 0.9881, 0.9884, 0.9925]
primerPerdidaEntrenamiento = [0.7887, 0.5785, 0.4991, 0.4518, 0.4360, 0.4135, 0.3945, 0.3583, 0.3647, 0.3464, 0.3331, 0.3096, 0.2836, 0.2958, 0.2876, 0.2661, 0.2565, 0.2357, 0.2375, 0.2178, 0.2202, 0.1941, 0.1862, 0.1722, 0.1723, 0.1561, 0.1524, 0.1442, 0.1301, 0.1179, 0.1088, 0.1002, 0.0969, 0.0822, 0.0871, 0.0805, 0.0723, 0.0648, 0.0808, 0.0525, 0.0540, 0.0462, 0.0359, 0.0426, 0.0379, 0.0388, 0.0375, 0.0431, 0.0329, 0.0250]
primerAciertoValidacion = [0.4493, 0.8104, 0.8119, 0.8299, 0.8731, 0.8269, 0.8776, 0.8761, 0.8836, 0.8701, 0.8791, 0.8896, 0.8881, 0.8866, 0.8701, 0.9000, 0.8836, 0.8955, 0.8687, 0.8776, 0.8940, 0.8597, 0.9045, 0.9030, 0.8940, 0.9075, 0.9015, 0.8806, 0.9254, 0.8955, 0.7746, 0.9134, 0.9015, 0.8776, 0.9239, 0.9119, 0.9030, 0.8090, 0.9104, 0.8776, 0.8881, 0.9075, 0.8687, 0.9030, 0.8881, 0.8970, 0.8836, 0.9104, 0.8836, 0.9045]
primerPerdidaValidacion = [0.7718, 0.4443, 0.4352, 0.4107, 0.3402, 0.4245, 0.3506, 0.3321, 0.3061, 0.3416, 0.3160, 0.2912, 0.2907, 0.3077, 0.3649, 0.2807, 0.3507, 0.2879, 0.4035, 0.3803, 0.2914, 0.5472, 0.2745, 0.2940, 0.3333, 0.2806, 0.2755, 0.4460, 0.2746, 0.3612, 0.4871, 0.2981, 0.4093, 0.4792, 0.2778, 0.3407, 0.3186, 1.4196, 0.3275, 0.3822, 0.5217, 0.3962, 0.6441, 0.4328, 0.5382, 0.5696, 0.4228, 0.3685, 0.5860, 0.4647]

epochs = []

for i in range(50):

    epochs.append(i + 1)

plt.title("Precisión Modelo")
plt.plot(epochs, primerAciertoEntrenamiento, label = "Entrenamiento", color = "blue")
plt.plot(epochs, primerAciertoValidacion, label = "Validación", color = "red")
plt.axhline(y = max(primerAciertoEntrenamiento), linestyle = "--", color = "darkblue")
plt.axhline(y = max(primerAciertoValidacion), linestyle = "--", color = "darkred")
plt.xlabel("Epochs")
plt.ylabel("Porcentaje")
plt.legend()
plt.show()

plt.title("Error Modelo")
plt.plot(epochs, primerPerdidaEntrenamiento, label = "Entrenamiento", color = "blue")
plt.plot(epochs, primerPerdidaValidacion, label = "Validación", color = "red")
plt.axhline(y = min(primerPerdidaEntrenamiento), linestyle = "--", color = "darkblue")
plt.axhline(y = min(primerPerdidaValidacion), linestyle = "--", color = "darkred")
plt.xlabel("Epochs")
plt.ylabel("Error")
plt.legend()
plt.show()
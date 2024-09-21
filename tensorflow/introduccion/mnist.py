import tensorflow as tf

# Sera evaluado el dataset mnist remplazando los modelos estimator por las versiones más recientes ofrecidas por tensorflow haciendo uso de keras.

mnist = tf.keras.datasets.mnist

# Desempaquetar los datos en mis variables de entrenamiento y testeo

(xTrain, yTrain), (xTest, yTest) = mnist.load_data()

# Normalizar mis datos en lugar de reducir sus dimensiones a 1 bit de color tomares los 8 pero en un rango flotante.

xTrain, xTest = xTrain / 255.0, xTest / 255.0

# Sera desarrollado un modelo secuencial útil para stackear capas cuando se espera disponer un tensor de entrada y uno de salida.
# Son suceptibles de elegir capaz flatten, Dense y Droput.

# Flatten: Transforma una matriz en un vector unidimensional
# Dense: Explorado en las redes hechas desde cero una neurona esta conectada con todas las siguientes mediante un camino denominado peso.
# Droput: Deshabilita varias neuronas durante el entrenamiento a fin de evtiar overfitting, dispone de un indice denominado drop rate.
# El drop rate dictamina la probabilidad de deshabilitar una neurona en el entrenamiento, evita la presencia de neuronas con alta dependencia, pero relantiza el entrenamiento.

model = tf.keras.models.Sequential([

    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation = "relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)

])

predictions = model(xTrain[:1]).numpy() # Realiza un slice del tensor evalua solamente el primer caso.
print("Predicciones: \n{}".format(predictions))

# Aplicar la funcion de activacion softmax en la capa de predicciones eleva una exponencial al dato brindado y divido por una sumatoria de potencias.
# Es usada softmax en clasificación múltiple ya que considera el output de todas las neuronas y les asigna una distribución de probabilidad.

tf.nn.softmax(predictions).numpy()

# Aplicar una función de perdida para evaluar el error obtenido.

lossFn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)

model.compile(optimizer = "adam",
              loss = lossFn,
              metrics = ["accuracy"])

model.fit(xTrain, yTrain, epochs = 5)

# Verbose condiciona como son mostrados los datos, 1 incorpora una barra de progreso y 2 la omite.

model.evaluate(xTest, yTest, verbose = 2)
import pandas as pd
import tensorflow as tf

# Compatible con versiones infererios a 2.3 Tensorflow.

# Cargar los datos desde una api de Google, referente al titanic, cargarlo de esta manera agrega funcionalidades extras que almacernarlo en un array númerico.

dataFrameTrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
dataFrameEvaluar = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')

print("Primer Data Frame \n\n{}".format(dataFrameTrain.head()))

# Remover la columna survived y almacenar los resultados en una variable.

yTrain = dataFrameTrain.pop("survived")
YEvaluar = dataFrameEvaluar.pop("survived")

#yTrain.iloc[Indice] Muestra la información almacenada como si fuera un arrray.

print("\n\nInformación general del dataframe\n\n")

print(dataFrameEvaluar.describe())

#dataFrame shape muestra las dimensiones de los datos.
#print(dataFrameTrain.age.hist(bins= 20))

# A continuación sera construido un modelo de regresión múltiple para predecir la supervivencia de un pasajero del Titanic.
# Primero requiere establecer los criterios de evalauación y preprar los datos, es separado las columnas númericos con las cadenas.

categoriasColumnas = ["sex", "n_siblings_spouses", "parch", "class", "deck", "embark_town", "alone"]
categoriasNumericas = ["Age", "Fare"]

# Guaradara todas los elementos unicos de cada columna

vectorCaracteristico = []

for featureName in categoriasColumnas:

    vocabulario = dataFrameTrain[featureName].unique() # El dataFrame de Pandas retorna todos los valores.

    # Agregame los valores únicos.
    
    # La funcion categorical_column_with_vocabulary_list se encuentra fuera de servicio para versiones recientes de tf.
    
    vectorCaracteristico.append(tf.feature_column.categorical_column_with_vocabulary_list(featureName, vocabulario))
    
for featureName in categoriasNumericas:

    vectorCaracteristico.append(tf.feature_column.categorical_column_with_vocabulary_list(featureName, dtype=tf.float32))

# Los datos deben ser preparados para su entrenamiento y predicción dispone primordialmente de dos parametros epochs y batchSize.
# Epoch: Número de veces que el modelo evaluara la misma información, son las generaciones de entrenamiento conforme más sean aumenta la precisión a coste de generar overfitting o sobreajuste.
# batchSize: Número de elementos por Lote, en algunas circunstancias la cantidad de datos superara la capacidad de alojarlos en memoria, por ende son segmentados.

def inputFn(dataFrame, labelDataFrame, epochs = 10, shuffle = True, batchSize = 32):

    def inputFunction():

        # Crea un tensor donde es indicado la pertencia de múltiples valores a una etiqueta
        ds = tf.data.Dataset.from_tensor_slices((dict(dataFrame), labelDataFrame))

        if shuffle:
            ds = ds.shuffle(1000)

        ds = ds.batch(batchSize).repeat(epochs)
        return ds
    
    return inputFunction

datosPreparadosEntrenamiento = inputFn(dataFrameTrain, yTrain) # yTrain seran mis valores de salida en unra red neuronal indicarian mis nodos de salida.
datosPreparadosEvaluar = inputFn(dataFrameEvaluar, YEvaluar, epochs = 1, shuffle = False) # No necesito evaluarlo más de 1 vez y tampoco la aleatoriedad.

# Una vez preparado los datos y agrupandolos bajo etiqutas unicas todas las caracteristicas es entrenado los modelos.

modeloLineal = tf.estimator.LinearClassifier(feature_columns = vectorCaracteristico)
modeloLineal.train(datosPreparadosEntrenamiento)
resultado = modeloLineal.evaluate(dataFrameEvaluar)

print(resultado["acurrancy"])
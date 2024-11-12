import tensorflow as tf
from tensorflow import keras
import numpy as np
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight # type: ignore
from tensorflow.keras.models import load_model # type: ignore

# Cargar el set de datos.

direccionDataset = Path("weedDetectionInWheat/Dataset")
plantas = list(direccionDataset.glob('train/docks/*'))
direccionEntrenamiento = direccionDataset / "train/"
direccionValidamiento = direccionDataset / "valid/"

# Especificar las dimensiones de las imagenes y el tamaño de lotes.

anchoImagen = 227
largoImagen = 227
imgSize = [anchoImagen, largoImagen]
batchSize = 32 

# Crear los dataframes.

trainDataFrame = tf.keras.utils.image_dataset_from_directory(

    direccionEntrenamiento,
    #validation_split = 0.2, Es recomendable colocar 0.2, aunque el dataset ya se encuentra separado.
    #subset = "training",
    seed = 123,
    image_size = imgSize,
    batch_size = batchSize,
    label_mode = "binary"

)

validacionDataFrame = tf.keras.utils.image_dataset_from_directory(

    direccionValidamiento,
    #validation_split = 0.2, Es recomendable colocar 0.2, aunque el dataset ya se encuentra separado.
    #subset = "training",
    seed = 123,
    image_size = imgSize,
    batch_size = batchSize,
    label_mode = "binary"

)

# Tenemos una mayor presencia de una clase respecto a otra en el dataset, por ende ajustamos los pesos de las clases acorde la presencia de datos.

etiquetasDataset = np.concatenate([y for x, y in trainDataFrame], axis = 0)

etiquetasDataset = etiquetasDataset.flatten()

pesosClases = compute_class_weight(class_weight = "balanced",
                                   classes = np.unique(etiquetasDataset),
                                   y = etiquetasDataset)

pesosClasesDiccionario = {}
clasesUnicas = np.unique(etiquetasDataset)

for i in range(len(clasesUnicas)):
    pesosClasesDiccionario[int(clasesUnicas[i])] = float(pesosClases[i])

# Aplicar Data Argumentation en el modelo a fin de incrementar la cantidad de datos de entrenamiento.

dataArgumentation = tf.keras.Sequential([

    # Transformaciones Geometricas

    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomTranslation(0.1, 0.1),

    # Transformaciones Color       

    tf.keras.layers.RandomContrast(0.2),           # Ajuste del contraste
    tf.keras.layers.RandomBrightness(0.2),         # Ajuste del brillo                               
                                          
])

def procesarImagen(x, y):

    return dataArgumentation(x), y

# Modificar el dataset.

dataArgumentationTrain = trainDataFrame.map(procesarImagen)

# Después de cada convolución es normalizado los datos e incorporado un maxpool para abstraer las características más predominantes.

alexnet = keras.models.Sequential([

    keras.layers.Input(shape=(227, 227, 3)),

    # Primera capa convolucional de 96 Kernels de (11, 11)

    keras.layers.Conv2D(filters = 96, kernel_size = (11, 11),
                        strides = (4, 4), activation = "relu",
                        kernel_initializer = "he_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size = (3, 3), strides = (2, 2),
                           padding = "valid", data_format = None),

    # Segunda capa convolucional de 256 Kernels de (5, 5)

    keras.layers.Conv2D(filters = 256, kernel_size = (5, 5),
                        strides = (1, 1), activation = "relu", padding = "same",
                        kernel_initializer = "he_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size = (3, 3), strides = (2, 2), 
                           padding = "valid", data_format = None),

    # Tercer capa convolucional de 384 Kernels de (3, 3)

    keras.layers.Conv2D(filters = 384, kernel_size = (3, 3),
                        strides = (1, 1), activation = "relu", padding = "same",
                        kernel_initializer = "he_normal"),
    keras.layers.BatchNormalization(),

    # Cuarta capa convolucional de 384 Kernels (1, 1)

    keras.layers.Conv2D(filters = 384, kernel_size = (1, 1),
                        strides = (1, 1), activation = "relu", padding = "same",
                        kernel_initializer = "he_normal"),
    keras.layers.BatchNormalization(),

    # Quinta capa convolucional de 256 Kernels (1, 1)

    keras.layers.Conv2D(filters = 256, kernel_size = (1, 1),
                        strides = (1, 1), activation = "relu", padding = "same",
                        kernel_initializer = "he_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size = (3, 3), strides = (2, 2), 
                           padding = "valid",data_format = None),

    keras.layers.Flatten(),
    keras.layers.Dense(4096, activation = "relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(4096, activation = "relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1000, activation = "relu"),
    keras.layers.Dense(1, activation = "sigmoid") # Cambiamos la última capa de salida por una neurona y la función de activación sigmoid.
    
])

alexnet.compile(

    loss='binary_crossentropy', # Es cambiado el método númerico de perdida, al disponer de un resultado binario es optado binary crossentropy.
    optimizer=tf.keras.optimizers.Adam(0.001),
    metrics=['accuracy'] 
)

alexnet.summary()

history=alexnet.fit(
    dataArgumentationTrain,
    epochs=100,
    validation_data=validacionDataFrame,
    validation_freq=1,
    class_weight = pesosClasesDiccionario
)

# Almacenar el modelo en la siguiente dirección relativa.

alexnet.evaluate(validacionDataFrame, verbose = 1)

alexnet.save('weedDetectionInWheat/CNN/alexnet.keras') 

# Mostrar los datos relativos al entrenamiento.

alexnet.history.history.keys()

print('Accuracy Score = ',np.max(history.history['val_accuracy']))
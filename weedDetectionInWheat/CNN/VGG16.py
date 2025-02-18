import tensorflow as tf
from tensorflow import keras
import numpy as np
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight # type: ignore
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras import regularizers # type: ignore

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

VGG16 = keras.models.Sequential([

    keras.layers.Input(shape=(227, 227, 3)),

    keras.layers.Conv2D(64, (3, 3), padding="same", kernel_regularizer=regularizers.l2(0.0001)),
    keras.layers.BatchNormalization(),
    keras.layers.Activation("relu"),
    keras.layers.Conv2D(64, (3, 3), padding="same", kernel_regularizer=regularizers.l2(0.0001)),
    keras.layers.BatchNormalization(),
    keras.layers.Activation("relu"),
    keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same"),

    # Segunda capa convolucional
    keras.layers.Conv2D(128, (3, 3), padding="same", kernel_regularizer=regularizers.l2(0.0001)),
    keras.layers.BatchNormalization(),
    keras.layers.Activation("relu"),
    keras.layers.Conv2D(128, (3, 3), padding="same", kernel_regularizer=regularizers.l2(0.0001)),
    keras.layers.BatchNormalization(),
    keras.layers.Activation("relu"),
    keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same"),

    # Tercera capa convolucional
    keras.layers.Conv2D(256, (3, 3), padding="same", kernel_regularizer=regularizers.l2(0.0001)),
    keras.layers.BatchNormalization(),
    keras.layers.Activation("relu"),
    keras.layers.Conv2D(256, (3, 3), padding="same", kernel_regularizer=regularizers.l2(0.0001)),
    keras.layers.BatchNormalization(),
    keras.layers.Activation("relu"),
    keras.layers.Conv2D(256, (3, 3), padding="same", kernel_regularizer=regularizers.l2(0.0001)),
    keras.layers.BatchNormalization(),
    keras.layers.Activation("relu"),
    keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same"),

    # Cuarta capa convolucional
    keras.layers.Conv2D(512, (3, 3), padding="same", kernel_regularizer=regularizers.l2(0.0001)),
    keras.layers.BatchNormalization(),
    keras.layers.Activation("relu"),
    keras.layers.Conv2D(512, (3, 3), padding="same", kernel_regularizer=regularizers.l2(0.0001)),
    keras.layers.BatchNormalization(),
    keras.layers.Activation("relu"),
    keras.layers.Conv2D(512, (3, 3), padding="same", kernel_regularizer=regularizers.l2(0.0001)),
    keras.layers.BatchNormalization(),
    keras.layers.Activation("relu"),
    keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same"),

    # Quinta capa convolucional
    keras.layers.Conv2D(512, (3, 3), padding="same", kernel_regularizer=regularizers.l2(0.0001)),
    keras.layers.BatchNormalization(),
    keras.layers.Activation("relu"),
    keras.layers.Conv2D(512, (3, 3), padding="same", kernel_regularizer=regularizers.l2(0.0001)),
    keras.layers.BatchNormalization(),
    keras.layers.Activation("relu"),
    keras.layers.Conv2D(512, (3, 3), padding="same", kernel_regularizer=regularizers.l2(0.0001)),
    keras.layers.BatchNormalization(),
    keras.layers.Activation("relu"),
    keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same"),

    keras.layers.Flatten(),
    keras.layers.Dense(4096, activation = "relu", kernel_regularizer=regularizers.l2(0.0001)),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(4096, activation = "relu", kernel_regularizer=regularizers.l2(0.0001)),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation = "sigmoid") # Cambiamos la última capa de salida por una neurona y la función de activación sigmoid.
    
])

VGG16.compile(

    loss='binary_crossentropy', # Es cambiado el método númerico de perdida, al disponer de un resultado binario es optado binary crossentropy.
    optimizer=tf.keras.optimizers.Adam(0.001),
    metrics=['accuracy'] 
)

VGG16.summary()

history=VGG16.fit(
    dataArgumentationTrain,
    epochs=100,
    validation_data=validacionDataFrame,
    validation_freq=1,
    class_weight = pesosClasesDiccionario
)

# Almacenar el modelo en la siguiente dirección relativa.

VGG16.evaluate(validacionDataFrame, verbose = 1)

VGG16.save('weedDetectionInWheat/CNN/VGG16.keras') 

# Mostrar los datos relativos al entrenamiento.

VGG16.history.history.keys()

print('Accuracy Score = ',np.max(history.history['val_accuracy']))
import tensorflow as tf
from tensorflow import keras
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Cargar el set de datos.

direccionDataset = Path("weedDetectionInWheat/Docknet")
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
    batch_size = batchSize

)

validacionDataFrame = tf.keras.utils.image_dataset_from_directory(

    direccionValidamiento,
    #validation_split = 0.2, Es recomendable colocar 0.2, aunque el dataset ya se encuentra separado.
    #subset = "training",
    seed = 123,
    image_size = imgSize,
    batch_size = batchSize

)

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

    keras.layers.Conv2D(filters = 384, kernel_size = (1, 1),
                        strides = (1, 1), activation = "relu", padding = "same",
                        kernel_initializer = "he_normal"),
    keras.layers.BatchNormalization(),

    keras.layers.Conv2D(filters = 256, kernel_size = (1, 1),
                        strides = (1, 1), activation = "relu", padding = "same",
                        kernel_initializer = "he_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size = (3, 3), strides = (2, 2), 
                           padding = "valid",data_format = None),

    keras.layers.Flatten(),
    keras.layers.Dense(4096, activation = "relu"),
    keras.layers.Dense(4096, activation = "relu"),
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
    trainDataFrame,
    epochs=20,
    validation_data=validacionDataFrame,
    validation_freq=1
)

# Almacenar el modelo en la siguiente dirección relativa.

alexnet.save('weedDetectionInWheat/alexnet.keras') 

# Mostrar los datos relativos al entrenamiento.

alexnet.history.history.keys()

f,ax = plt.subplots(2,1,figsize=(10,10)) 

ax[0].plot(alexnet.history.history['loss'],color='b',label='Training Loss')
ax[0].plot(alexnet.history.history['val_loss'],color='r',label='Validation Loss')

#Plotting the training accuracy and validation accuracy
ax[1].plot(alexnet.history.history['accuracy'],color='b',label='Training  Accuracy')
ax[1].plot(alexnet.history.history['val_accuracy'],color='r',label='Validation Accuracy')

print('Accuracy Score = ',np.max(history.history['val_accuracy']))

plt.legend()
plt.show()
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

anchoImagen = 224
largoImagen = 224
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

# Versión reducida de Alexnet.

alexnet = keras.models.Sequential([

    keras.layers.Input(shape=(224, 224, 3)),

    keras.layers.Conv2D(filters = 128, kernel_size = (11, 11),
                        strides = (4, 4), activation = "relu"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size = (2, 2)),

    keras.layers.Conv2D(filters = 256, kernel_size = (5, 5),
                        strides = (1, 1), activation = "relu", padding = "same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size = (3, 3)),

    keras.layers.Conv2D(filters = 256, kernel_size = (3, 3),
                        strides = (1, 1), activation = "relu", padding = "same"),
    keras.layers.BatchNormalization(),

    keras.layers.Conv2D(filters = 256, kernel_size = (1, 1),
                        strides = (1, 1), activation = "relu", padding = "same"),
    keras.layers.BatchNormalization(),

    keras.layers.Conv2D(filters = 256, kernel_size = (1, 1),
                        strides = (1, 1), activation = "relu", padding = "same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size = (2, 2)),

    keras.layers.Flatten(),
    keras.layers.Dense(1024, activation = "relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1024, activation = "relu"),
    keras.layers.Dropout(0.5), 
    keras.layers.Dense(1, activation = "sigmoid") # Cambiamos la última capa de salida por una neurona y la función de activación sigmoid.
    
])

alexnet.compile(

    loss='binary_crossentropy', # Es cambiado el método númerico de perdida, al disponer de un resultado binario es optado binary crossentropy.
    optimizer=tf.optimizers.SGD(learning_rate=0.001),
    metrics=['accuracy']    

)

alexnet.summary()

history=alexnet.fit(
    trainDataFrame,
    epochs=50,
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
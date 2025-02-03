import sys
import os
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
from keras.models import Model

# Añadir la carpeta necesaria para importaqr la clase alexnet

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from weedDetectionInWheat.CNN.alexnetClass import alexnet
from weedDetectionInWheat.metaheuristic.GWOGPU import GWO

direccionDataset = Path("weedDetectionInWheat/Dataset")
direccionEntrenamiento = direccionDataset / "train/"
direccionValidamiento = direccionDataset / "valid/"

anchoImagen = 227
largoImagen = 227
imgSize = [anchoImagen, largoImagen]
batchSize = 32

trainDataFrame = tf.keras.utils.image_dataset_from_directory(
    direccionEntrenamiento,
    seed=123,
    image_size=imgSize,
    batch_size=batchSize,
    label_mode="binary"
)

validacionDataFrame = tf.keras.utils.image_dataset_from_directory(
    direccionValidamiento,
    seed=123,
    image_size=imgSize,
    batch_size=batchSize,
    label_mode="binary"
)

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

# Construir el modelo AlexNet
CNN = alexnet(trainDataFrame, validacionDataFrame)
arquitecturaCNN = CNN.obtenerModelo()
pesosClases = CNN.obtenerPesosClases()

# Compilar y entrenar el modelo
arquitecturaCNN.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.Adam(0.001),
    metrics=['accuracy'],
)

alexnet_metaheuristic = keras.models.load_model("weedDetectionInWheat/CNN/alexnetMetaheuristic.keras")
alexnetGradiente = keras.models.load_model("weedDetectionInWheat/CNN/alexnetNormalized.keras")

def create_model(transfer_learning = None):
    for layer in alexnet_metaheuristic.layers:
            layer.trainable = False

    x = alexnet_metaheuristic.get_layer(name="mask").output

    if(transfer_learning is None):
        x = keras.layers.Dense(4096, activation="relu", kernel_regularizer=regularizers.l2(0.0001))(x)
        x = keras.layers.Dropout(0.5)(x)
        x = keras.layers.Dense(4096, activation="relu", kernel_regularizer=regularizers.l2(0.0001))(x)
        x = keras.layers.Dropout(0.5)(x)
        x = keras.layers.Dense(1000, activation="relu", kernel_regularizer=regularizers.l2(0.0001))(x)
        x = keras.layers.Dense(1, activation="sigmoid")(x)

    else:

        flatten_layer = alexnetGradiente.get_layer(name="flatten")
        flatten_index = alexnetGradiente.layers.index(flatten_layer)
        gradiente_layers = alexnetGradiente.layers[flatten_index + 1:]

        for layer in gradiente_layers:
            x = layer(x) 

    new_model = Model(inputs=alexnet_metaheuristic.input, outputs=x)
    new_model.summary()
    new_model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(0.0001),
        metrics=['accuracy']
    )

    return new_model

new_model = create_model()

history = new_model.fit(
    dataArgumentationTrain,
    epochs=100,
    validation_data=validacionDataFrame,
    validation_freq=1,
    class_weight = pesosClases
)

with open("metaheuristic.json", "w") as f:
    json.dump(history.history, f)

new_model.save("weedDetectionInWheat/CNN/alexnet_combined.keras")

new_model = create_model(transfer_learning=True)

history = new_model.fit(
    dataArgumentationTrain,
    epochs=100,
    validation_data=validacionDataFrame,
    validation_freq=1,
    class_weight = pesosClases
)

with open("metaheuristic_transfer_learning.json", "w") as f:
    json.dump(history.history, f)

new_model.save("weedDetectionInWheat/CNN/alexnet_combined_transfer_learning.keras")
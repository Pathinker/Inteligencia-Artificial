import tensorflow as tf
from tensorflow import keras
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight # type: ignore

class alexnet():

    def __init__(self, datasetEntrenamiento, datasetValidacion, nombre = None, epochs = None, train = None):
        
        self.alexnet = self.crearModelo()
        self.weights = self.balancerPesos(datasetEntrenamiento)
        self.dataArgumentation = self.generarDataArgumentation(datasetEntrenamiento)

        if(train != None):

            self.fit(self.dataArgumentation, datasetValidacion, nombre, epochs)
            self.datasetEntrenamiento = datasetEntrenamiento

    def crearModelo(self):

        modelo = keras.models.Sequential([

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

        modelo.compile(

            loss='binary_crossentropy', # Es cambiado el método númerico de perdida, al disponer de un resultado binario es optado binary crossentropy.
            optimizer=tf.keras.optimizers.Adam(0.001),
            metrics=['accuracy']    

        )

        modelo.summary()

        return modelo
    
    def obtenerModelo(self):
        
        return self.alexnet
    
    def obtenerPesosClases(self):
        
        return self.weights
    
    def balancerPesos(self, datasetEntrenamiento):

        etiquetasDataset = np.concatenate([y for x, y in datasetEntrenamiento], axis = 0)

        etiquetasDataset = etiquetasDataset.flatten()

        pesosClases = compute_class_weight(class_weight = "balanced",
                                        classes = np.unique(etiquetasDataset),
                                        y = etiquetasDataset)

        pesosClasesDiccionario = {}
        clasesUnicas = np.unique(etiquetasDataset)

        for i in range(len(clasesUnicas)):
            pesosClasesDiccionario[int(clasesUnicas[i])] = float(pesosClases[i])

        return pesosClasesDiccionario
    
    def generarDataArgumentation(self, datasetEntrenamiento):

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

        dataArgumentationTrain = datasetEntrenamiento.map(procesarImagen)

        return dataArgumentationTrain

    def fit(self, datasetEntrenamiento, datasetValidacion, nombre, epochs):

        if epochs == None:
            epochs = 50

        if nombre == None:
            nombre = "alexnet.keras"

        history = self.alexnet.fit(
        datasetEntrenamiento,
        epochs = epochs,
        validation_data=datasetValidacion,
        validation_freq=1,
        class_weight = self.weights
        )

        self.alexnet.save(f'weedDetectionInWheat/CNN/{nombre}') 

        self.alexnet.history.history.keys()

        f,ax = plt.subplots(2,1,figsize=(10,10)) 

        ax[0].plot(self.alexnet.history.history['loss'],color='b',label='Training Loss')
        ax[0].plot(self.alexnet.history.history['val_loss'],color='r',label='Validation Loss')

        #Plotting the training accuracy and validation accuracy
        ax[1].plot(self.alexnet.history.history['accuracy'],color='b',label='Training  Accuracy')
        ax[1].plot(self.alexnet.history.history['val_accuracy'],color='r',label='Validation Accuracy')

        print('Accuracy Score = ',np.max(history.history['val_accuracy']))
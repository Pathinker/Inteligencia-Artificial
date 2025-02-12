import tensorflow as tf
from tensorflow import keras
import numpy as np
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight # type: ignore

from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras import regularizers # type: ignore
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization
from tensorflow.keras.layers import MaxPool2D, GlobalAvgPool2D, Flatten
from tensorflow.keras.layers import Add, ReLU, Dense
from tensorflow.keras import Model

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
    seed = 123,
    image_size = imgSize,
    batch_size = batchSize,
    label_mode = "binary"

)

validacionDataFrame = tf.keras.utils.image_dataset_from_directory(

    direccionValidamiento,
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

def build_resnet101():

    def conv_batchnorm_relu(x, filters, kernel_size, strides=1):
        x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding = 'same', kernel_regularizer=regularizers.l2(0.0001))(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        return x

    def identity_block(tensor, filters):
        x = conv_batchnorm_relu(tensor, filters=filters, kernel_size=1, strides=1)
        x = conv_batchnorm_relu(x, filters=filters, kernel_size=3, strides=1)
        x = Conv2D(filters=4*filters, kernel_size=1, strides=1, kernel_regularizer=regularizers.l2(0.0001))(x)
        x = BatchNormalization()(x)
        x = Add()([tensor,x])
        x = ReLU()(x)
        return x

    def projection_block(tensor, filters, strides): 
                
        x = conv_batchnorm_relu(tensor, filters=filters, kernel_size=1, strides=strides)     
        x = conv_batchnorm_relu(x, filters=filters, kernel_size=3, strides=1)     
        x = Conv2D(filters=4*filters, kernel_size=1, strides=1, kernel_regularizer=regularizers.l2(0.0001))(x)     
        x = BatchNormalization()(x) 
                
        shortcut = Conv2D(filters=4*filters, kernel_size=1, strides=strides)(tensor)     
        shortcut = BatchNormalization()(shortcut)          
        x = Add()([shortcut,x])     
        x = ReLU()(x)          
        return x

    def resnet_block(x, filters, reps, strides):
        
        x = projection_block(x, filters, strides)
        for _ in range(reps-1):
            x = identity_block(x,filters)
        return x

    input = Input(shape=(227,227,3))
    x = conv_batchnorm_relu(input, filters=64, kernel_size=7, strides=2)
    x = MaxPool2D(pool_size = 3, strides =2)(x)
    x = resnet_block(x, filters=64, reps =3, strides=1)
    x = resnet_block(x, filters=128, reps =4, strides=2)
    x = resnet_block(x, filters=256, reps =23, strides=2)
    x = resnet_block(x, filters=512, reps =3, strides=2)
    x = GlobalAvgPool2D()(x)
    x = Flatten()(x)
    x = Dense(1000, activation = "relu", kernel_regularizer=regularizers.l2(0.0001))(x)
    output = Dense(1, activation = "sigmoid")(x)

    resnet101 = Model(inputs=input, outputs=output)
    
    return resnet101

resnet101 = build_resnet101()

resnet101.compile(

    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.Adam(0.001),
    metrics=['accuracy'] 
)

resnet101.summary()

history=resnet101.fit(
    dataArgumentationTrain,
    epochs=100,
    validation_data=validacionDataFrame,
    validation_freq=1,
    class_weight = pesosClasesDiccionario
)

resnet101.evaluate(validacionDataFrame, verbose = 1)
resnet101.save('weedDetectionInWheat/CNN/resnet101.keras')
resnet101.history.history.keys()
print('Accuracy Score = ',np.max(history.history['val_accuracy']))
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras import regularizers # type: ignore
from sklearn.utils.class_weight import compute_class_weight # type: ignore

from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras import regularizers # type: ignore
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization
from tensorflow.keras.layers import MaxPool2D, GlobalAvgPool2D, Flatten
from tensorflow.keras.layers import Add, ReLU, Dense
from tensorflow.keras import Model

dataset_path = Path("weedDetectionInWheat/Dataset")
train_path = dataset_path / "train/"
validation_path = dataset_path / "valid/"

anchoImagen = 227
largoImagen = 227
image_size = [anchoImagen, largoImagen]
batch_size = 24

train_data_frame = tf.keras.utils.image_dataset_from_directory(
    train_path,
    seed=123,
    image_size=image_size,
    batch_size=batch_size,
    label_mode="binary"
)

validation_data_frame = tf.keras.utils.image_dataset_from_directory(
    validation_path,
    seed=123,
    image_size=image_size,
    batch_size=batch_size,
    label_mode="binary"
)

data_argumentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
    tf.keras.layers.RandomContrast(0.2),
    tf.keras.layers.RandomBrightness(0.2),
])

def apply_data_argumentation(data_frame):

    def process_image(x, y):
        return data_argumentation(x), y

    data_argumentation = data_frame.map(process_image)
    labels = np.concatenate([y for x, y in train_data_frame], axis = 0)
    labels = labels.flatten()

    return data_argumentation, labels

def balance_clases_dataset(labels):

    class_weight = compute_class_weight(class_weight = "balanced",
                                    classes = np.unique(labels),
                                    y = labels)
    unique_classes = np.unique(labels)
    weights = {}

    for i in range(len(unique_classes)):
        weights[int(unique_classes[i])] = float(class_weight[i])

    return weights

train_data_argumentation, labels = apply_data_argumentation(train_data_frame)
class_weights =  balance_clases_dataset(labels)

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
    train_data_argumentation,
    epochs=100,
    validation_data=validation_data_frame,
    validation_freq=1,
    class_weight = class_weights
)

resnet101.evaluate(validation_data_frame, verbose = 1)
resnet101.save('weedDetectionInWheat/CNN/resnet101.keras')

resnet101.history.history.keys()
print('Best validation accuracy score = ',np.max(history.history['val_accuracy']))
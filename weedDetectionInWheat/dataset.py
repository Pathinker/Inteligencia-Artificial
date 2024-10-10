import pandas as pd
import numpy as np
import PIL
import PIL.Image
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf

# Normalizar el dataset de prueba y entrenamiento.

direccionDataset = Path("weedDetectionInWheat/Dataset")

# Busqueda profunda en subsecuentes niveles haciendo uso de **, * solamente busca en la carpeta inmediata.

numeroImagenes = len(list(direccionDataset.glob('**/*.jpg')))

# Almacenar en un array todas las fotografias.

plantas = list(direccionDataset.glob('train/docks/*'))
direccionEntrenamiento = direccionDataset / "train/"
direccionValidamiento = direccionDataset / "valid/"

# Mostrar dos imagenes.

img = PIL.Image.open(str(plantas[0]))
img.show()

img = PIL.Image.open(str(plantas[1]))
img.show()

# Generar los dataframes de entrenaimiento 

anchoImagen = 224
largoImagen = 224
imgSize = [anchoImagen, largoImagen]

batchSize = 32 

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

# Visualizar el nombre de las clases que encontro el modelo.

nombreClases = trainDataFrame.class_names

print(nombreClases)

# Mostrar 10 figuras del dataset

plt.figure(figsize=(10, 10))

for images, labels in trainDataFrame.take(1):
  
  for i in range(9):

    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(nombreClases[labels[i]])
    plt.axis("off")

plt.show()
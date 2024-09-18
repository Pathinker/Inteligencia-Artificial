import tensorflow as tf

# Tensorflow hace uso de de tensores, los caules son vectores o matrices con una gran cantidad de dimensiones.
# Construye un graph of Tensor para dar resultados, cada tensor o capa tiene su propio tipo de dato y dimensiones
# Cada graph realiza una computación parcial del resultado final, al momento de incorporar un nuevo valor solamente indica la operación a realizar, no evalua el resultado.
# Son denominados grafos al estar correlacionados todos los datos realizando las operaciones según sea requerido.

# Una sesión ejecuta por completo o partes del grafo, por ejemplo las primeras capaz con valores independientes y constantes.

#  Creación de tensores 

tensorString = tf.Variable("Hola Mundo", tf.string)
tensorNumero = tf.Variable(324, tf.int16) # Longitud de bits del int de 2^16 - 1
tensorFloat = tf.Variable(3.2212, tf.float64) # Longitud de bits del float de 2 ^ 64 - 1

# Tensores escalares por tener unicamente una variable

primerDimension = tf.Variable(["Tensor de una dimension al tener solamente un elemento"], tf.string)
segundaDimension = tf.Variable([["Soy la ", "Primera Dimension"],["Soy la", "Segunda Dimension"]], tf.string)

# Siempre necesito tener matrices cuadradas.

print("\nDimensiones de una variable o tensores: {}".format(tf.rank(segundaDimension)))

# La dimensión va acorde al número mayor de arreglos anidados.

primerDimension.shape # Número de elementos de cada dimensión.

# 1 = Número de Lotes o Depth (Profundidad)
# 2 = Número de Filas (Cantidad de Filas por Lote, cantidad de arrays)
# 3 = Número de Columnas (Cantida de Columnas, datos por arrays)

tensor1 = tf.ones([1, 2, 3]) # 1 Lista con 2 Arrays y 3 Elementos por Array
tensor2 = tf.reshape(tensor1, [2, 3, 1]) # 2 Listas con 3 Arrays y 1 Elemento por Array.
tensor3 = tf.reshape(tensor2, [3, -1]) # 3 Arrays con 2 Elementos, el -1 calcula directamente la cantidad la cantidad.

# El reshape deben coincidir la cantidad de elementos.

print("\nPrimer Tensor: {} ".format(tensor1))
print("\nSegundo Tensor: {}".format(tensor2))
print("\nTercer Tensor: {}".format(tensor3))

# Crea una sesión la cual ejecuta una parte parcial del grafo generado por tensorflow.
# Como se cito al inicio tensorflow elabora grafos los cuales realizan computos parciales hasta dar con el resultado.

# with tf.Session() as sess:
   # tensor1.eval()
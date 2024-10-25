import numpy as np
import tensorflow as tf
from tqdm import tqdm

class ADSCFGWO:
    def __init__(self, model, iterMaximo=50, classWeight = None):

        self.model = model
        self.iterMaximo = iterMaximo
        self.classWeight = classWeight

        # Obtener los pesos iniciales del modelo
        self.weights_structure = model.get_weights()

        # Inicializamos las posiciones de los lobos en función de la estructura de los pesos del modelo

        self.positions = self.asignarPosicion()

    def asignarPosicion(self): # Generar una matriz con todos los pesos a optimizar de la red.

        pocisionRandom = []

        # Iterar sobre cada conjunto de pesos en la estructura de pesos del modelo
        for w in self.weights_structure:

            # Generar una matriz de valores aleatorios con la misma forma que los pesos 'w'
            random_weights = np.random.uniform(-1, 1, w.shape)
            pocisionRandom.append(random_weights)

        return pocisionRandom
    
    def setWeights(self, weights):

        self.model.set_weights(weights)

    def calcularFitness(self, loss):
        
        # Calcula la función de fitness basada en la pérdida del modelo y el número de características seleccionadas.
        
        alfa = 0.99
        beta = 0.01

        # Fitness basado en la ecuación Hn = α * Error(P) + β * |S|/|A|
        fitness = alfa * loss + beta * 1

        return fitness
    
    def busquedaDinamica(self, perdidas, verificarValores):
         
         # Verificar que almenos existen almenos los valores consecutivos que se desean evaluar

         if(len(perdidas) >= verificarValores):

            # Establecer un valor de referencia para evaluar valore consecutivos en caso de ser igual retornamos True para incrementar el coeficiente de exploracion.

            auxiliar = perdidas[len(perdidas) - verificarValores]

            for i in range(1, (verificarValores - 1)):

                if(perdidas[len(perdidas) - i ] != auxiliar):

                    return False

            return True
         
         else:
             
             return False
        
    def calcularPerdidaConPesos(self, dataset, classWeight, epoch = None):

        loss = 0
        total = 0
        prediccionesCorrectas = 0
        
        for x, y in tqdm(dataset, desc = f"Epoch {epoch} / {self.iterMaximo}", unit="batch"):

            prediccion = self.model.predict(x, verbose = 0)  # Realizar la predicción

            lossBatch = tf.keras.losses.binary_crossentropy(y, prediccion)

            # Convertir y a un arreglo unidimensional

            etiqueta = y.numpy().flatten()  # Asegúrate de que sea un vector 1D

            # Aplicar los pesos de clase

            if classWeight is not None:

                weights = np.zeros_like(lossBatch.numpy())  # Crear un arreglo de ceros con la misma forma que loss_batch
                
                for i in range(len(etiqueta)): 
                    label = etiqueta[i]
                    weights[i] = classWeight[label]         

                lossBatch *=  weights

            #print(lossBatch)
            #lossBatch = tf.clip_by_value(lossBatch, clip_value_min=1e-7, clip_value_max=1.0)
            
            loss += tf.reduce_sum(lossBatch).numpy()  # Sumar la pérdida del batch
            total += len(y)

            prediccionClase = tf.round(prediccion)
            prediccionesCorrectas += tf.reduce_sum(tf.cast(tf.equal(prediccionClase, y), tf.float32)).numpy()  # Contar aciertos transformando un tensor ft float 32.
        
        accuracy = prediccionesCorrectas / total
        print(f"Precisión: {accuracy} Pérdida: {loss / total}")

        return loss / total  # Retornar la pérdida promedio    

    def optimize(self, datasetEntrenamiento, datasetEvaluacion):

        # Inicializar alpha, beta y delta (los mejores lobos)
        posicionAlfa, posicionBeta, posicionDelta = self.positions, self.positions, self.positions
        lossAlfa, lossBeta, lossDelta =  np.inf, np.inf, np.inf

        losses = []
        coeficienteExploracion = 0.0
        
        # Asignar los pesos del lobo actual al modelo
        self.setWeights(self.positions)

        # Evaluar la pérdida en los datos de entrenamiento
        print("Pesos asignados aleatoriamente: ")
        loss = self.calcularPerdidaConPesos(datasetEntrenamiento, self.classWeight, 0)

        for iteracion in range(self.iterMaximo):

            incrementarExploracion = self.busquedaDinamica(losses, 3)

            if(incrementarExploracion):

                if(coeficienteExploracion == 0.0):

                    coeficienteExploracion += 0.1

                coeficienteExploracion *= 1.05

            # Definir los valores por los cuales seran dictaminadas las dos variables constantes para la exploración y explotacion.

            r1 = coeficienteExploracion + np.random.random() * (1 - coeficienteExploracion)
            r2 = coeficienteExploracion + np.random.random() * (1 - coeficienteExploracion)
            a = 2 - iteracion * (2/self.iterMaximo)

            r1SCA = a - (a * iteracion / self.iterMaximo)
            r2SCA = coeficienteExploracion + np.random.random() * (1 - coeficienteExploracion)
            r3SCA = coeficienteExploracion + np.random.random() * (1 - coeficienteExploracion)
            r4SCA = np.random.random() # Determina el uso de coseno y seno, no alterar

            A = 2 * a * r1 - a
            C = 2 * r2

            # Calcular fitness
            fitness = self.calcularFitness(loss)

            # Actualizar alpha, beta y delta al detectar un agente menor equivalente a una menor perdida

            if iteracion == 0 :
                
                lossDelta, posicionDelta = fitness, self.positions.copy()
                lossBeta, posicionBeta = fitness, self.positions.copy()
                lossAlfa, posicionAlfa = fitness, self.positions.copy()

            if fitness < lossAlfa:

                # Actualizar Alpha y mover los otros lobos hacia abajo
                lossDelta, posicionDelta = lossBeta, posicionBeta.copy()
                lossBeta, posicionBeta = lossAlfa, posicionAlfa.copy()
                lossAlfa, posicionAlfa = fitness, self.positions.copy()

            elif fitness < lossBeta:

                # Actualizar Beta y mover Delta hacia abajo
                lossDelta, posicionDelta = lossBeta, posicionBeta.copy()
                lossBeta, posicionBeta = fitness, self.positions.copy()

            elif fitness < lossDelta:
                    
                # Actualizar solo Delta
                lossDelta, posicionDelta = fitness, self.positions.copy()

            # Normalizar las pérdidas

            perdidaTotal = lossAlfa + lossBeta + lossDelta

            if perdidaTotal > 0:  

                lossAlfa /= perdidaTotal
                lossBeta /= perdidaTotal
                lossDelta /= perdidaTotal                 

            # Actualizar las posiciones de los lobos

            distanciaAlfa = None
            distanciaBeta = None
            distanciaDelta = None

            for i in range(len(self.weights_structure)):

                # Calculo de la distancia del lobo a la presa.

                r1SCA = a - (a * iteracion / self.iterMaximo)
                r2SCA = np.random.random()
                r3SCA = np.random.random()
                r4SCA = np.random.random()

                M = np.abs(C * (lossAlfa * posicionAlfa[i] + lossBeta * posicionBeta[i] + lossDelta * posicionDelta[i]) - self.positions[i])

                # Reposionamiento del lobo.

                self.positions[i] = (
                    (posicionAlfa[i] - A * M +
                    posicionBeta[i] - A * M +
                    posicionDelta[i] - A * M) / 3
                )

                # Efectuar el algoritmo ASA 

                r1SCA = a - (a * iteracion / self.iterMaximo)

                if r4SCA < 0.5:
                    self.positions[i] += (
                    r1SCA * np.sin(r2SCA) * np.abs(r3SCA * posicionAlfa[i] - self.positions[i])
                    )
                elif r4SCA >= 0.5:
                    self.positions[i] += (
                    r1SCA * np.cos(r2SCA) * np.abs(r3SCA * posicionAlfa[i] - self.positions[i])
                    )
            
            # Asignar los pesos del lobo actual al modelo

            self.setWeights(self.positions)

            # Evaluar la pérdida en los datos de entrenamiento

            loss = self.calcularPerdidaConPesos(datasetEntrenamiento, self.classWeight, iteracion + 1)
            losses.append(loss)

            # Evaluar la pérdida en los datos de evaluación

            self.model.evaluate(datasetEvaluacion, verbose=1)

        # Devuelve los mejores pesos encontrados
        return posicionAlfa
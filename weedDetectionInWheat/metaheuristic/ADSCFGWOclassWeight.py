import numpy as np
import tensorflow as tf
from tqdm import tqdm

class ADSCFGWO:
    def __init__(self, model, iterMaximo=10, numeroAgentes = 5, classWeight = None):

        # Hiperparametros del constructor

        self.model = model
        self.iterMaximo = iterMaximo
        self.numeroAgentes = numeroAgentes # Número de población, soluciones a buscar en cada iteración.
        self.classWeight = classWeight # Balanceo de clases
        self.coeficienteBusqueda = 0.0

        self.weights_structure = model.get_weights()
        self.cantidadPesos = self.obtenerCantidadPesos()

        # Variables GWO

        self.loss = []
        self.lossAlfa = np.finfo(np.float32).max
        self.lossBeta = np.finfo(np.float32).max
        self.lossDelta =  np.finfo(np.float32).max

        self.positions = np.zeros((numeroAgentes, self.cantidadPesos))
        self.posicionAlfa = np.zeros(self.cantidadPesos)
        self.posicionBeta = np.zeros(self.cantidadPesos)
        self.posicionDelta = np.zeros(self.cantidadPesos)

        # Obtener los pesos iniciales del modelo con el objetivo de obtener su estructura

        for i in range(self.numeroAgentes):

            self.positions[i] = self.asignarPosicion()
        
        self.setWeights(self.positions[0])

    def obtenerCantidadPesos(self):
        
        pesosTotales = 0

        for w in self.weights_structure:

            elementos = np.prod(w.shape)  # Producto de todas las dimensiones de la forma
            pesosTotales += elementos

        return pesosTotales

    def asignarPosicion(self): # Generar una matriz con todos los pesos a optimizar de la red.

        pocisionRandom = []

        # Iterar sobre cada conjunto de pesos en la estructura de pesos del modelo
        for w in self.weights_structure:

            # Generar una matriz de valores aleatorios con la misma forma que los pesos 'w'
            random_weights = np.random.uniform(-1, 1, w.shape)
            pocisionRandom.append(random_weights.flatten())

        return np.concatenate(pocisionRandom)
    
    def setWeights(self, weights):

        new_weights = []
        idx = 0
        for w in self.weights_structure:
            
            shape = w.shape
            size = np.prod(shape)
            new_weights.append(np.array(weights[idx:idx + size]).reshape(w.shape)) # Asignar a la forma original
            idx += size

        self.model.set_weights(new_weights)

    def calcularPerdidaConPesos(self, dataset, classWeight):

        prediccionesCorrectas = 0
        total = 0
        loss = 0
        
        for x, y in tqdm(dataset, desc = f"Calculando Perdida", unit="batch"):

            # Realizar una predicción por batch y extraer su perdida.

            prediccion = self.model.predict(x, verbose = 0) 
            lossBatch = tf.keras.losses.binary_crossentropy(y, prediccion)
            etiqueta = y.numpy().flatten()  

            if classWeight is not None:

                weights = np.zeros_like(lossBatch.numpy())  # Crear un arreglo de ceros con la misma forma que loss_batch
                
                for i in range(len(etiqueta)): 
                    label = etiqueta[i]
                    weights[i] = classWeight[label]    

                lossBatch *=  weights
            
            loss += tf.reduce_sum(lossBatch).numpy()  # Sumar la pérdida del batch
            total += len(y)

            prediccionClase = tf.round(prediccion)
            prediccionesCorrectas += tf.reduce_sum(tf.cast(tf.equal(prediccionClase, y), tf.float32)).numpy()  # Contar aciertos transformando un tensor ft float 32.
        
        accuracy = prediccionesCorrectas / total
        print(f"Precisión: {accuracy} Pérdida: {loss / total}")

        return loss / total  

    def calcularFitness(self, loss):
        
        # Calcula la función de fitness basada en la pérdida del modelo y el número de características seleccionadas.
        
        alfa = 0.99
        beta = 0.01

        # Fitness basado en la ecuación Hn = α * Error(P) + β * |S|/|A|
        fitness = alfa * loss + beta * 1

        return fitness
    
    def busquedaDinamica(self, perdidas, verificarValores):
         
         # Verificar que almenos existen almenos los valores consecutivos que se desean evaluar

        if (len(perdidas) < verificarValores):

            return False

        # Establecer un valor de referencia para evaluar valore consecutivos en caso de ser igual retornamos True para incrementar el coeficiente de exploracion.

        auxiliar = perdidas[len(perdidas) - verificarValores]

        for i in range(1, (verificarValores - 1)):

            if(perdidas[len(perdidas) - i ] != auxiliar):

                return False

        return True
         
    def optimize(self, datasetEntrenamiento, datasetEvaluacion):

        print("Ajustar Parametros Aleatorios: ")

        loss = self.calcularPerdidaConPesos(datasetEntrenamiento, self.classWeight)
        self.loss.append(loss)
        fitness = self.calcularFitness(loss)

        self.lossAlfa, self.posicionAlfa = fitness, np.ravel(self.positions[0, :].copy())
        self.lossBeta, self.posicionBeta = fitness, np.ravel(self.positions[0, :].copy())
        self.lossDelta, self.posicionDelta = fitness, np.ravel(self.positions[0, :].copy())

        for iteracion in range(self.iterMaximo):

            self.GWO(datasetEntrenamiento, datasetEvaluacion, iteracion, 0)
            self.GWO(datasetEntrenamiento, datasetEvaluacion, iteracion, 1)

        return self.posicionAlfa
    
    def GWO(self, datasetEntrenamiento, datasetEvaluacion, iteracion, trigonometrica):

        for n in range(self.numeroAgentes):

            # Evaluar la pérdida en los datos de entrenamiento

            print(f"Epoch {iteracion + 1} / {self.iterMaximo} (Poblacion {trigonometrica + 1}, Agente {n + 1} / {self.numeroAgentes})| Entrenamiento | Validación: ")

            loss = self.calcularPerdidaConPesos(datasetEntrenamiento, self.classWeight)
            self.loss.append(loss)

            fitness = self.calcularFitness(loss)

            # Evaluar la pérdida en los datos de evaluación

            self.model.evaluate(datasetEvaluacion, verbose=1)    

            if(self.busquedaDinamica(self.loss, 3)):

                self.coeficienteBusqueda *= 1.05

                if(self.coeficienteBusqueda == 0.0):

                    self.coeficienteBusqueda = 0.1

            r1 =  self.coeficienteBusqueda + (np.random.random() * (1 -  self.coeficienteBusqueda))
            r2 =  self.coeficienteBusqueda + (np.random.random() * (1 -  self.coeficienteBusqueda))
            r3 =  self.coeficienteBusqueda + (np.random.random() * (1 -  self.coeficienteBusqueda))

            r4 = np.random.random()
            a = 2 - iteracion * (2/self.iterMaximo)

            r1SCA = a - (a * iteracion / self.iterMaximo)

            A = 2 * a * r1 - a
            C = 2 * r2

            # Actualizar alpha, beta y delta al detectar un agente menor equivalente a una menor perdida

            print("Error: ", fitness)

            if fitness < self.lossAlfa:

                print("Actualización Alfa: ", self.lossAlfa)
                print("Posiciones Alfa: ", self.posicionAlfa)

                # Actualizar Alpha y mover los otros lobos hacia abajo
                
                self.lossDelta, self.posicionDelta = self.lossBeta, np.ravel(self.posicionBeta.copy())
                self.lossBeta, self.posicionBeta = self.lossAlfa, np.ravel(self.posicionAlfa.copy())
                self.lossAlfa, self.posicionAlfa = fitness, np.ravel(self.positions[n, :].copy())

            elif fitness < self.lossBeta:

                print("Actualización Beta: ", self.lossBeta)
                print("Posiciones Beta: ", self.posicionBeta)

                self.lossDelta, self.posicionDelta = self.lossBeta, np.ravel(self.posicionBeta.copy())
                self.lossBeta, self.posicionBeta = fitness, np.ravel(self.positions[n, :].copy())

            elif fitness < self.lossDelta:

                print("Actualización Delta: ", self.lossDelta)
                print("Posiciones Delta: ", self)
                    
                self.lossDelta, self.posicionDelta = fitness, np.ravel(self.positions[n, :].copy())

            print("Perdida Alfa: ", self.lossAlfa)
            print("Posiciones Alfa: ", self.posicionAlfa)
            print("Perdida Beta: ", self.lossBeta)
            print("Posiciones Beta: ", self.posicionBeta)
            print("Perdida Delta: ", self.lossDelta)
            print("Posiciones Delta: ", self.posicionDelta)

            # Normalizar las pérdidas

            # perdidaTotal = self.lossAlfa + self.lossBeta + self.lossDelta

            #if perdidaTotal > 0:  

                #self.lossAlfa /= perdidaTotal
                #self.lossBeta /= perdidaTotal
                #self.lossDelta /= perdidaTotal
            
            # Actualizar las posiciones de los lobos     

            for i in tqdm(range(len(self.positions[n])), desc=f"Ajustando pesos", unit="peso"):

                # Calculo de la distancia del lobo a la presa.

                posicionAlfa = self.posicionAlfa[i]
                posicionBeta = self.posicionBeta[i]
                posicionDelta = self.posicionDelta[i]
                positions = self.positions[n][i]

                M = np.abs(C * (self.lossAlfa * posicionAlfa + 
                                self.lossBeta * posicionBeta + 
                                self.lossDelta * posicionDelta) - positions)
                                
                V1 = self.posicionAlfa[i] - A * M
                V2 = self.posicionBeta[i] - A * M 
                V3 = self.posicionDelta[i] - A * M

                # Reposionamiento del lobo.

                self.positions[n][i] = (V1 + V2 + V3) / 3

                # Efectuar el algoritmo ASA 

                if (trigonometrica == 0 and r4 < 0.5):

                    self.positions[n][i] += (
                    r1SCA * np.sin(r2) * np.abs(r3 * self.posicionAlfa[i] - self.positions[n][i])
                    )

                elif (trigonometrica == 1 and r4 >= 0.5):

                    self.positions[n][i] += (
                    r1SCA * np.cos(r2) * np.abs(r3 * self.posicionAlfa[i] - self.positions[n][i])
                    )

            self.setWeights(self.positions[n])
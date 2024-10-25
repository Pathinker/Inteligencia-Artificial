import numpy as np
import tensorflow as tf
from tqdm import tqdm

class ADSCFGWO:
    def __init__(self, model, iterMaximo=10, numeroAgentes = 5, classWeight = None):

        # Hiperparametros del constructor

        self.model = model # Modelo usado de la red.
        self.iterMaximo = iterMaximo # Iteraciones máximas a realizar.
        self.numeroAgentes = numeroAgentes # Número de población, soluciones a buscar en cada iteración.
        self.classWeight = classWeight # Balanceo de clases
        self.coeficienteBusqueda = 0.0

        # Variables GWO

        self.loss = []

        self.lossAlfa = np.inf
        self.lossBeta = np.inf
        self.lossDelta =  np.inf

        self.posicionAlfa = None
        self.posicionBeta = None
        self.posicionDelta = None

        # Obtener los pesos iniciales del modelo

        self.weights_structure = model.get_weights()

        # Inicializamos las posiciones de los lobos en función de la estructura de los pesos del modelo

        self.positions = self.asignarPosicion()
        self.setWeights(self.positions)

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

        return loss / total  # Retornar la pérdida promedio    

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

        # Inicializar alpha, beta y delta (los mejores lobos)

        self.posicionAlfa, self.posicionBeta, self.posicionDelta = self.positions, self.positions, self.positions

        # Evaluar la pérdida en los datos de entrenamiento

        print("Pesos asignados aleatoriamente:")
        
        loss = self.calcularPerdidaConPesos(datasetEntrenamiento, self.classWeight)
        fitness = self.calcularFitness(loss)

        for iteracion in range(self.iterMaximo):

            if iteracion == 0 :
                
                self.lossDelta, self.posicionDelta = fitness, self.positions.copy()
                self.lossBeta, self.posicionBeta = fitness, self.positions.copy()
                self.lossAlfa, self.posicionAlfa = fitness, self.positions.copy()

            self.GWO(datasetEntrenamiento, datasetEvaluacion, iteracion, 0)
            self.GWO(datasetEntrenamiento, datasetEvaluacion, iteracion, 1)

        # Devuelve los mejores pesos encontrados
        return self.posicionAlfa
    
    def GWO(self, datasetEntrenamiento, datasetEvaluacion, iteracion, trigonometrica):

        fitness = np.inf

        for n in range(self.numeroAgentes):

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

            if fitness < self.lossAlfa:

                # Actualizar Alpha y mover los otros lobos hacia abajo
                self.lossDelta, self.posicionDelta = self.lossBeta, self.posicionBeta.copy()
                self.lossBeta, self.posicionBeta = self.lossAlfa, self.posicionAlfa.copy()
                self.lossAlfa, self.posicionAlfa = fitness, self.positions.copy()

            elif fitness < self.lossBeta:

                # Actualizar Beta y mover Delta hacia abajo
                self.lossDelta, self.posicionDelta = self.lossBeta, self.posicionBeta.copy()
                self.lossBeta, self.posicionBeta = fitness, self.positions.copy()

            elif fitness < self.lossDelta:
                    
                # Actualizar solo Delta
                self.lossDelta, self.posicionDelta = fitness, self.positions.copy()

            # Normalizar las pérdidas

            perdidaTotal = self.lossAlfa + self.lossBeta + self.lossDelta

            if perdidaTotal > 0:  

                self.lossAlfa /= perdidaTotal
                self.lossBeta /= perdidaTotal
                self.lossDelta /= perdidaTotal       

            # Actualizar las posiciones de los lobos     

            for i in range(len(self.weights_structure)):

                # Calculo de la distancia del lobo a la presa.

                M = np.abs(C * (self.lossAlfa * self.posicionAlfa[i] + 
                                self.lossBeta * self.posicionBeta[i] + 
                                self.lossDelta * self.posicionDelta[i]) - self.positions[i])
                
                V1 = self.posicionAlfa[i] - A * M
                V2 = self.posicionBeta[i] - A * M 
                V3 = self.posicionDelta[i] - A * M

                # Reposionamiento del lobo.

                self.positions[i] = ((V1 + V2 + V3)/3)

                # Efectuar el algoritmo ASA 

                if (trigonometrica == 0 and r4 < 0.5):

                    self.positions[i] += (
                    r1SCA * np.sin(r2) * np.abs(r3 * self.posicionAlfa[i] - self.positions[i])
                    )

                elif (trigonometrica == 1 and r4 >= 0.5):

                    self.positions[i] += (
                    r1SCA * np.cos(r2) * np.abs(r3 * self.posicionAlfa[i] - self.positions[i])
                    )
            
            # Asignar los pesos del lobo actual al modelo

            self.setWeights(self.positions)

            # Evaluar la pérdida en los datos de entrenamiento

            print(f"Epoch {iteracion + 1} / {self.iterMaximo} (Poblacion {trigonometrica + 1}, Agente {n + 1} / {self.numeroAgentes})| Entrenamiento | Validación: ")

            loss = self.calcularPerdidaConPesos(datasetEntrenamiento, self.classWeight)
            self.loss.append(loss)

            fitness = self.calcularFitness(loss)

            # Evaluar la pérdida en los datos de evaluación

            self.model.evaluate(datasetEvaluacion, verbose=1)    
import numpy as np
import tensorflow as tf

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

    def calcularFitness(self, loss, weights):
        
        # Calcula la función de fitness basada en la pérdida del modelo y el número de características seleccionadas.
        
        alfa = 0.99
        beta = 0.01

        # Fitness basado en la ecuación Hn = α * Error(P) + β * |S|/|A|
        fitness = alfa * loss + beta * 1

        return fitness
    
    def calcularPerdida(self, parametro):

        if(np.isinf(parametro)):
            return 0
        
        return parametro
    
    def normalizarPerdida(self, parametro, total):

        if(np.isinf(parametro)):
            return parametro
        
        return parametro / total

    def optimize(self, datasetEntrenamiento, datasetEvaluacion):

        # Inicializar alpha, beta y delta (los mejores lobos)
        posicionAlfa, posicionBeta, posicionDelta = self.positions, self.positions, self.positions
        lossAlfa, lossBeta, lossDelta =  np.inf, np.inf, np.inf

        # Asignar los pesos del lobo actual al modelo
        self.setWeights(self.positions)

        # Evaluar la pérdida en los datos de entrenamiento
        print("Pesos asignados aleatoriamente: ")
        loss, _ = self.model.evaluate(datasetEntrenamiento, verbose = 1)

        for iteracion in range(self.iterMaximo):

            # Definir los valores por los cuales seran dictaminadas las dos variables constantes para la exploración y explotacion.

            r1 = np.random.random()
            r2 = np.random.random()
            r3 = np.random.random()
            r4 = np.random.random()
            a = 2 - iteracion * (2/self.iterMaximo)

            A = 2 * a * r1 - a
            C = 2 * r2

            # Calcular fitness
            fitness = self.calcularFitness(loss, self.positions)

            # Actualizar alpha, beta y delta al detectar un agente menor equivalente a una menor perdida

            if iteracion == 0 :
                
                lossDelta, posicionDelta = fitness, self.positions.copy()
                lossBeta, posicionBeta = fitness, self.positions.copy()
                lossAlfa, posicionAlfa = fitness, self.positions.copy()

            if fitness < lossAlfa:

                # Actualizar Alpha y mover los otros lobos hacia abajo
                lossDelta, posicionDelta = lossBeta, posicionBeta
                lossBeta, posicionBeta = lossAlfa, posicionAlfa
                lossAlfa, posicionAlfa = fitness, self.positions.copy()

            elif fitness < lossBeta:

                # Actualizar Beta y mover Delta hacia abajo
                lossDelta, posicionDelta = lossBeta, posicionBeta
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

                M = np.abs(C * (lossAlfa * posicionAlfa[i] + lossBeta * posicionBeta[i] + lossDelta * posicionDelta[i]) - self.positions[i])

                # Reposionamiento del lobo.

                self.positions[i] = (
                    (posicionAlfa[i] - A * M +
                    posicionBeta[i] - A * M +
                    posicionDelta[i] - A * M) / 3
                )

                # Efectuar el algoritmo ASA 

                r1SCA = a - (a * iteracion / self.iterMaximo)

                if r4 < 0.5:
                    self.positions[i] += (
                    r1SCA * np.sin(r2) * np.abs(r3 * posicionAlfa[i] - self.positions[i])
                    )
                elif r4 >= 0.5:
                    self.positions[i] += (
                    r1SCA * np.cos(r2) * np.abs(r3 * posicionAlfa[i] - self.positions[i])
                    )
            
            # Asignar los pesos del lobo actual al modelo

            self.setWeights(self.positions)

            # Evaluar la pérdida en los datos de entrenamiento

            print(f"Epoch {iteracion + 1} / {self.iterMaximo} Entrenamiento / Validación: ")

            loss, _ = self.model.evaluate(datasetEntrenamiento, verbose = 1)

            # Evaluar la pérdida en los datos de evaluación

            self.model.evaluate(datasetEvaluacion, verbose=1)

        # Devuelve los mejores pesos encontrados
        return posicionAlfa
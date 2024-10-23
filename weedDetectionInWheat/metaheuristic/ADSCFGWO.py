import numpy as np
import tensorflow as tf

class ADSCFGWO:
    def __init__(self, numeroAgentes, model, iterMaximo=50):

        self.numeroAgentes = numeroAgentes
        self.model = model
        self.iterMaximo = iterMaximo

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
    
    def optimize(self, datasetEntrenamiento, datasetEvaluacion):

        # Inicializar alpha, beta y delta (los mejores lobos)
        posicionAlfa, posicionBeta, posicionDelta = self.positions, self.positions, self.positions
        lossAlfa, lossBeta, lossDelta = np.inf, np.inf, np.inf

        for iteracion in range(self.iterMaximo):

            # Definir los valores por los cuales seran dictaminadas las dos variables constantes para la exploración y explotacion.

            r1 = np.random.random()
            r2 = np.random.random()
            r3 = np.random.random()
            r4 = np.random.random()
            a = 2 - iteracion * (2/self.iterMaximo)

            A = 2 * a * r1 - a
            C = 2 * r2

            # Asignar los pesos del lobo actual al modelo
            self.setWeights(self.positions)

            # Evaluar la pérdida en los datos de entrenamiento
            loss, _ = self.model.evaluate(datasetEntrenamiento, verbose = 1)

            # Evaluar la pérdida en los datos de evaluación

            loss_eval = self.model.evaluate(datasetEvaluacion, verbose=1)

            # Actualizar alpha, beta y delta al detectar un agente menor equivalente a una menor perdida

            if loss < lossAlfa:

                # Actualizar Alpha y mover los otros lobos hacia abajo
                lossDelta, posicionDelta = lossBeta, posicionBeta
                lossBeta, posicionBeta = lossAlfa, posicionAlfa
                lossAlfa, posicionAlfa = loss, self.positions

            elif loss < lossBeta:

                # Actualizar Beta y mover Delta hacia abajo
                lossDelta, posicionDelta = lossBeta, posicionBeta
                lossBeta, posicionBeta = loss, self.positions

            elif loss < lossDelta:
                    
                # Actualizar solo Delta
                lossDelta, posicionDelta = loss, self.positions

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

                # M = np.abs(C * (lossAlfa * distanciaAlfa, ))

                distanciaAlfa = np.abs(C * posicionAlfa[i] - self.positions[i])
                distanciaBeta = np.abs(C * posicionBeta[i] - self.positions[i])
                distanciaDelta = np.abs(C * posicionDelta[i] - self.positions[i]) 

                # Reposionamiento del lobo.

                self.positions[i] = (
                    (posicionAlfa[i] - A * distanciaAlfa +
                    posicionBeta[i] - A * distanciaBeta +
                    posicionDelta[i] - A *distanciaDelta) / 3
                )

                # Efectuar el algoritmo ASA 

                if r4 < 0.5:
                    self.positions[i] += (
                    r1 * np.sin(r2) * np.abs(r3 * posicionAlfa[i] - self.positions[i])
                    )
                elif r4 >= 0.5:
                    self.positions[i] += (
                    r1 * np.cos(r2) * np.abs(r3 * posicionAlfa[i] - self.positions[i])
                    )

        # Devuelve los mejores pesos encontrados
        return posicionAlfa
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

        self.positions = []

        for i in range(self.numeroAgentes):

            position = self.asignarPosicion()
            self.positions.append(position)

    def asignarPosicion(self):

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
        posicionAlfa, posicionBeta, posicionDelta = None, None, None
        resultadoAlfa, resultadoBeta, resultadoDelta = np.inf, np.inf, np.inf

        for iteracion in range(self.iterMaximo):

            # Definir los valores por los cuales seran dictaminadas las dos variables constantes para la exploración y explotacion.

            r1 = np.random.random()
            r2 = np.random.random()
            r3 = np.random.random()
            r4 = np.random.random()
            a = 2 - iteracion * (2/self.iterMaximo)

            A = 2 * a * r1 - a
            C = 2 * r2

            for i in range(self.numeroAgentes):

                # Asignar los pesos del lobo actual al modelo
                self.setWeights(self.positions[i])

                # Evaluar la pérdida en los datos de entrenamiento
                loss, _ = self.model.evaluate(datasetEntrenamiento, verbose = 1)

                # Evaluar la pérdida en los datos de evaluación

                loss_eval = self.model.evaluate(datasetEvaluacion, verbose=1)

                # Actualizar alpha, beta y delta
                if loss < resultadoAlfa:
                    resultadoAlfa, posicionAlfa = loss, self.positions[i]
                elif loss < resultadoBeta:
                    resultadoBeta, posicionBeta = loss, self.positions[i]
                elif loss < resultadoDelta:
                    resultadoDelta, posicionDelta = loss, self.positions[i]

                # Normalizar las pérdidas

                perdidaTotal = resultadoAlfa + resultadoBeta + resultadoDelta

                if perdidaTotal > 0:  

                    resultadoAlfa /= perdidaTotal
                    resultadoBeta /= perdidaTotal
                    resultadoDelta /= perdidaTotal                    

            # Actualizar las posiciones de los lobos
            for i in range(self.numeroAgentes):

                distanciaAlfa = None
                distanciaBeta = None
                distanciaDelta = None

                for j in range(len(self.weights_structure)):

                    # Calculo de la distancia del lobo a la presa.

                    distanciaAlfa = np.abs(C * posicionAlfa[j] - self.positions[i][j])
                    distanciaBeta = np.abs(C * posicionBeta[j] - self.positions[i][j])
                    distanciaDelta = np.abs(C * posicionDelta[j] - self.positions[i][j]) 

                    # Reposionamiento del lobo.

                    self.positions[i][j] = (
                        (posicionAlfa[j] - A * distanciaAlfa +
                        posicionBeta[j] - A * distanciaBeta +
                        posicionDelta[j] - A *distanciaDelta) / 3
                    )

                    # Efectuar el algoritmo ASA 

                    if r4 < 0.5:
                        self.positions[i][j] += (
                            r1 * np.sin(r2) * np.abs(r3 * posicionAlfa[j] - self.positions[i][j])
                        )
                    elif r4 >= 0.5:
                        self.positions[i][j] += (
                            r1 * np.cos(r2) * np.abs(r3 * posicionAlfa[j] - self.positions[i][j])
                        )

        # Devuelve los mejores pesos encontrados
        return posicionAlfa
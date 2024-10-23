import numpy as np
import tensorflow as tf

class GWO:
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
    
    def optimize(self, datasetEntrenamiento, etiquetasEntrenamiento):

        # Inicializar alpha, beta y delta (los mejores lobos)
        posicionAlfa, posicionBeta, posicionDelta = None, None, None
        resultadoAlfa, resultadoBeta, resultadoDelta = np.inf, np.inf, np.inf

        for iteracion in range(self.iterMaximo):

            # Definir los valores por los cuales seran dictaminadas las dos variables constantes para la exploración y explotacion.

            r1 = np.random.random()
            r2 = np.random.random()
            a = 2 - iteracion * (2/self.iterMaximo)

            A = 2 * a * r1 - a
            C = 2 * r2

            for i in range(self.numeroAgentes):

                # Asignar los pesos del lobo actual al modelo
                self.setWeights(self.positions[i])

                # Evaluar la pérdida en los datos de entrenamiento
                loss, _ = self.model.evaluate(datasetEntrenamiento, etiquetasEntrenamiento, verbose = 1)

                # Actualizar alpha, beta y delta
                if loss < resultadoAlfa:
                    resultadoAlfa, posicionAlfa = loss, self.positions[i]
                elif loss < resultadoBeta:
                    resultadoBeta, posicionBeta = loss, self.positions[i]
                elif loss < resultadoDelta:
                    resultadoDelta, posicionDelta = loss, self.positions[i]

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

        # Devuelve los mejores pesos encontrados
        return posicionAlfa
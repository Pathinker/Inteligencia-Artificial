import numpy as np
import tensorflow as tf
from tqdm import tqdm
import time

import pycuda.autoinit # type: ignore
import pycuda.driver as cuda  # type: ignore
from pycuda.compiler import SourceModule # type: ignore

class GWO:
    def __init__(self, model, iterMaximo=10, numeroAgentes = 5, classWeight = None):

        # Hiperparametros del constructor

        self.model = model
        self.iterMaximo = iterMaximo
        self.numeroAgentes = numeroAgentes # Número de población, soluciones a buscar en cada iteración.
        self.classWeight = classWeight # Balanceo de clases
        self.limiteSuperior = 100
        self.limiteInferior = -100

        self.weights_structure = model.get_weights()
        self.cantidadPesos = self.obtenerCantidadPesos()
        self.cantidadPesos = np.uint32(self.cantidadPesos)

        # Variables GWO

        self.lossAlfa = np.finfo(np.float64).max
        self.lossBeta = np.finfo(np.float64).max
        self.lossDelta =  np.finfo(np.float64).max

        self.accuracyAlfa = 0.0
        self.accuracyBeta = 0.0
        self.accuracyDelta = 0.0

        self.valLossAlfa = np.finfo(np.float64).max
        self.valLossBeta = np.finfo(np.float64).max
        self.valLossDelta =  np.finfo(np.float64).max

        self.valAccuracyAlfa = 0.0
        self.valAccuracyBeta = 0.0
        self.valAccuracyDelta = 0.0

        self.lossModel = []
        self.accuracyModel = []
        self.valLossModel = []
        self.valAccuracyModel = []

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
            random_weights = np.random.uniform(self.limiteInferior, self.limiteSuperior, w.shape)
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

        loss = 0.0
        accuracy = 0.0
        total = 0
        
        for x, y in tqdm(dataset, desc = f"Calculando Perdida", unit="batch"):

            pesosPonderados = []

            for i in y:

                peso = classWeight[int(i)]
                pesosPonderados.append(peso)

            pesosPonderados = np.array(pesosPonderados)
            
            lossBatch, accuracyBatch = self.model.evaluate(x, y, sample_weight = pesosPonderados, verbose = 0)

            loss += lossBatch
            accuracy += accuracyBatch
            total += 1

        print(f"loss: {loss/total}, Accuracy = {accuracy / total}")
        return (loss / total), (accuracy / total)  
    
    def optimize(self, datasetEntrenamiento, datasetEvaluacion):

        for iteracion in range(self.iterMaximo):

            self.GWOExploracion(datasetEntrenamiento, datasetEvaluacion, iteracion)
            self.GWOExplotacion(iteracion)

            self.accuracyModel.append(self.accuracyAlfa)
            self.lossModel.append(self.lossAlfa)
            self.valAccuracyModel.append(self.valAccuracyAlfa)
            self.valLossModel.append(self.valLossAlfa)
        
        self.setWeights(self.posicionAlfa)

        with open('weedDetectionInWheat/CNN/MetaheuristicReport.txt', 'w') as f:

            f.write(','.join(map(str, self.accuracyModel)) + '\n') 
            f.write(','.join(map(str, self.lossModel)) + '\n') 
            f.write(','.join(map(str, self.valAccuracyModel)) + '\n') 
            f.write(','.join(map(str, self.valLossModel)) + '\n')

        return self.model
    
    def GWOExploracion(self, datasetEntrenamiento, datasetEvaluacion, iteracion):

        for n in range(self.numeroAgentes):

            print(f"Exploración Epoch {iteracion + 1} / {self.iterMaximo} (Agente {n + 1} / {self.numeroAgentes})| Entrenamiento | Validación: ")

            self.setWeights(self.positions[n])
            loss, accuracy = self.calcularPerdidaConPesos(datasetEntrenamiento, self.classWeight)
            valLoss, valAccuracy = self.model.evaluate(datasetEvaluacion, verbose=1)    

            if loss < self.lossAlfa:

                print("Actualización Alfa")
                
                self.lossDelta, self.accuracyDelta, self.posicionDelta = self.lossBeta, self.accuracyBeta, np.ravel(self.posicionBeta.copy())
                self.lossBeta, self.accuracyBeta, self.posicionBeta = self.lossAlfa, self.accuracyAlfa, np.ravel(self.posicionAlfa.copy())
                self.lossAlfa, self.accuracyAlfa, self.posicionAlfa = loss, accuracy, np.ravel(self.positions[n, :].copy())

                self.valLossDelta, self.valAccuracyDelta = self.valLossBeta, self.valAccuracyBeta
                self.valLossBeta, self.valAccuracyBeta = self.valLossAlfa, self.valAccuracyAlfa
                self.valLossAlfa, self.valAccuracyAlfa = valLoss, valAccuracy

            elif loss < self.lossBeta:

                print("Actualización Beta")

                self.lossDelta, self.accuracyDelta, self.posicionDelta = self.lossBeta, self.accuracyBeta, np.ravel(self.posicionBeta.copy())
                self.lossBeta, self.accuracyBeta, self.posicionBeta = loss, accuracy, np.ravel(self.positions[n, :].copy())

                self.valLossDelta, self.valAccuracyDelta = self.valLossBeta, self.valAccuracyBeta
                self.valLossBeta, self.valAccuracyBeta = valLoss, valAccuracy

            elif loss < self.lossDelta:

                print("Actualización Delta")
                    
                self.lossDelta, self.accuracyDelta, self.posicionDelta = loss, accuracy, np.ravel(self.positions[n, :].copy())

                self.valLossDelta, self.valAccuracyDelta = valLoss, valAccuracy

            print(f"Alfa -> Perdida: {self.lossAlfa}, Accuracy: {self.accuracyAlfa}, valLoss: {self.valLossAlfa}, valAccuracy: {self.valAccuracyAlfa}")
            print(f"Beta -> Perdida: {self.lossBeta}, Accuracy: {self.accuracyBeta}, valLoss: {self.valLossBeta}, valAccuracy: {self.valAccuracyBeta}")
            print(f"Delta -> Perdida: {self.lossDelta}, Accuracy: {self.accuracyDelta}, valLoss: {self.valLossDelta}, valAccuracy: {self.valAccuracyDelta}")

    def GWOExplotacion(self, iteracion):

        for i in range(self.numeroAgentes):

            tiempoInicial = time.time()
            pesos = len(self.positions[0])

            a = 2 - iteracion * (2 / self.iterMaximo)

            r1 = np.random.uniform(0, 1, size=(3 * pesos)).astype(np.float32)
            r2 = np.random.uniform(0, 1, size=(3 * pesos)).astype(np.float32)

            posiciones = np.array(self.positions[i], dtype=np.float32)
            posicionAlfa = np.array(self.posicionAlfa, dtype=np.float32)
            posicionBeta = np.array(self.posicionBeta, dtype=np.float32)
            posicionDelta = np.array(self.posicionDelta, dtype=np.float32)
            
            # Alojar en  GPU

            distanciaPosiciones = cuda.mem_alloc(posiciones.nbytes)
            cuda.memcpy_htod(distanciaPosiciones, posiciones)

            distanciaPosicionAlfa = cuda.mem_alloc(posicionAlfa.nbytes)
            cuda.memcpy_htod(distanciaPosicionAlfa, posicionAlfa)

            distanciaPosicionBeta = cuda.mem_alloc(posicionBeta.nbytes)
            cuda.memcpy_htod(distanciaPosicionBeta, posicionBeta)

            distanciaPoscionDelta = cuda.mem_alloc(posicionDelta.nbytes)
            cuda.memcpy_htod(distanciaPoscionDelta, posicionDelta)

            R1GPU = cuda.mem_alloc(r1.nbytes)
            R2GPU = cuda.mem_alloc(r2.nbytes)
            cuda.memcpy_htod(R1GPU, r1)
            cuda.memcpy_htod(R2GPU, r2)

            mod = SourceModule("""
            __global__ void actualizar(float *posiciones, float *posicionAlfa, float *posicionBeta, float *posicionDelta,
                                    float *r1, float *r2, float a, int numeroPesos, float limiteInferior, float limiteSuperior) {
                int thread = blockIdx.x * blockDim.x + threadIdx.x;
                
                if (thread < numeroPesos) {

                    float A1 = 2 * a * r1[thread] - a;
                    float C1 = 2 * r2[thread];
                    float A2 = 2 * a * r1[thread +  numeroPesos] - a;
                    float C2 = 2 * r2[thread + numeroPesos];
                    float A3 = 2 * a * r1[thread + 2  * numeroPesos] - a;
                    float C3 = 2 * r2[thread + 2 * numeroPesos];

                    float posicionAlfa_i = posicionAlfa[thread];
                    float posicionBeta_i = posicionBeta[thread];
                    float posicionDelta_i = posicionDelta[thread];
                    float posicionSolucion_i = posiciones[thread];

                    float distanciaAlfa = fabs(C1 * posicionAlfa_i - posicionSolucion_i);
                    float distanciaBeta = fabs(C2 * posicionBeta_i - posicionSolucion_i);
                    float distanciaDelta = fabs(C3 * posicionDelta_i - posicionSolucion_i);

                    float X1 = posicionAlfa_i - A1 * distanciaAlfa;
                    float X2 = posicionBeta_i - A2 * distanciaBeta;
                    float X3 = posicionDelta_i - A3 * distanciaDelta;

                    posiciones[thread] = (X1 + X2 + X3) / 3;
                               
                    if(posiciones[thread] < limiteInferior){

                        posiciones[thread] = limiteInferior;            
                    }
                    else if(posiciones[thread] > limiteSuperior){
                               
                        posiciones[thread] = limiteSuperior;
                    }
                               
                }
            }
            """)

            # Inicializar y ejecutar el kernel
            actualizar_posiciones = mod.get_function("actualizar")
            block = 1024
            grid = (pesos + block - 1) // block

            actualizar_posiciones(distanciaPosiciones, distanciaPosicionAlfa, distanciaPosicionBeta, distanciaPoscionDelta,
                                R1GPU, R2GPU, np.float32(a), np.int32(pesos), np.float32(self.limiteInferior), np.float32(self.limiteSuperior),
                                block=(block, 1, 1), grid=(grid, 1))

            # Recuperamos los datos desde la GPU
            cuda.memcpy_dtoh(posiciones, distanciaPosiciones)
            self.positions[i] = posiciones

            tiempoFinal = time.time()
            print(f"Explotación {i + 1} / {self.numeroAgentes} tiempo de ejecución: {tiempoFinal - tiempoInicial} segundos")
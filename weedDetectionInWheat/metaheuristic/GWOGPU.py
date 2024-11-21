import numpy as np
import tensorflow as tf
from tqdm import tqdm
import time

import pycuda.autoinit # type: ignore
import pycuda.driver as cuda  # type: ignore
from pycuda.compiler import SourceModule # type: ignore

class GWO:
    def __init__(self, model, iterMaximo=10, numeroAgentes = 5,  numeroLobos = 5, classWeight = None, transferLearning = None):

        # Hiperparametros del constructor

        self.model = model
        self.iterMaximo = iterMaximo
        self.numeroAgentes = numeroAgentes # Número de población, soluciones a buscar en cada iteración.
        self.numeroLobos = numeroLobos
        self.classWeight = classWeight # Balanceo de clases
        self.transferLearning = transferLearning
        self.limiteSuperior = 1
        self.limiteInferior = -1

        self.weights_structure = model.get_weights()
        self.cantidadPesos = self.obtenerCantidadPesos()
        self.cantidadPesos = np.uint32(self.cantidadPesos)

        # Variables GWO

        self.nombreLobos = {0 : "Alfa", 1 : "Beta", 2: "Delta"}

        for i in range(self.numeroLobos - 3):

            self.nombreLobos[i + 3] = (f"Omega {i + 1}")

        self.loss = np.full(self.numeroLobos, np.finfo(np.float64).max)
        self.accuracy = []
        self.valLoss = np.full(self.numeroLobos, np.finfo(np.float64).max)
        self.valAccuracy = []
        
        self.lossModelLog = np.zeros((self.numeroLobos, self.iterMaximo))
        self.accuracyModelLog = np.zeros((self.numeroLobos, self.iterMaximo))
        self.valLossModelLog = np.zeros((self.numeroLobos, self.iterMaximo))
        self.valAccuracyModelLog = np.zeros((self.numeroLobos, self.iterMaximo))
        
        for i in range(self.numeroLobos):

            self.accuracy.append(0.0)
            self.valAccuracy.append(0.0)

        self.positions = np.zeros((numeroAgentes, self.cantidadPesos))
        self.positionsLobos = np.zeros((self.numeroLobos, self.cantidadPesos))

        if(transferLearning is None):

            for i in range(self.numeroAgentes):

                self.positions[i] = self.asignarPosicion()
        
        else:

            self.asignarLearning()

        self.setWeights(self.positions[0])

    def obtenerCantidadPesos(self):
        
        pesosTotales = 0

        for w in self.weights_structure:

            elementos = np.prod(w.shape)  # Producto de todas las dimensiones de la forma
            pesosTotales += elementos

        return pesosTotales

    def asignarPosicion(self): # Generar una matriz con todos los pesos a optimizar de la red.

        pocisionRandom = []

        for w in self.weights_structure:

            # Generar una matriz de valores aleatorios con la misma forma que los pesos 'w'
            random_weights = np.random.uniform(self.limiteInferior, self.limiteSuperior, w.shape)
            pocisionRandom.append(random_weights.flatten())

        return np.concatenate(pocisionRandom)

    def asignarLearning(self):

        self.transferLearning= self.model.get_weights()
        flattenedWeights = np.concatenate([weight.flatten() for weight in self.transferLearning])
        self.transferLearning = flattenedWeights

        for i in range(self.numeroAgentes):

            tiempoInicial = time.time()
            pesos = len(self.transferLearning)

            posiciones = np.array(self.positions[i], dtype=np.float32)
            transferLearning = np.array(self.transferLearning, dtype=np.float32)

            distanciaPosiciones = cuda.mem_alloc(posiciones.nbytes)
            cuda.memcpy_htod(distanciaPosiciones, posiciones)

            transferLearning = cuda.mem_alloc(self.transferLearning.nbytes)
            cuda.memcpy_htod(transferLearning, self.transferLearning)

            mod = SourceModule("""

            __device__ float xorshift32(unsigned int seed) {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                return (seed & 0x7FFFFFFF) / float(0x7FFFFFFF); // Normalizar a rango [0, 1]
            }
            __global__ void actualizar(float *posiciones, float *transferLearning, int numeroPesos,
                                        float limiteInferior, float limiteSuperior,  unsigned int seed) {
                int thread = blockIdx.x * blockDim.x + threadIdx.x;
                
                if (thread < numeroPesos) {
                               
                    unsigned int hilo_seed = seed + thread;
                    float random = xorshift32(hilo_seed);
            
                    posiciones[thread] = transferLearning[thread] * random;
                               
                    if(posiciones[thread] < limiteInferior){

                        posiciones[thread] = limiteInferior;            
                    }
                    else if(posiciones[thread] > limiteSuperior){
                               
                        posiciones[thread] = limiteSuperior;
                    }               
                }
            }
            """)

            actualizar_posiciones = mod.get_function("actualizar")
            block = 1024
            grid = (pesos + block - 1) // block
            seed = np.uint32(int(time.time() * 1000) % (2**32))

            actualizar_posiciones(distanciaPosiciones, transferLearning, np.int32(pesos), np.float32(self.limiteInferior), 
                                np.float32(self.limiteSuperior), seed, block=(block, 1, 1), grid=(grid, 1))

            cuda.memcpy_dtoh(posiciones, distanciaPosiciones)
            self.positions[i] = posiciones

            tiempoFinal = time.time()
            print(f"Inicialización {i + 1} / {self.numeroAgentes} tiempo de ejecución: {tiempoFinal - tiempoInicial} segundos")

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

        for epoch in range(self.iterMaximo):

            self.GWOExploracion(datasetEntrenamiento, datasetEvaluacion, epoch)
            self.GWOExplotacion(epoch)

            for i in range(self.numeroLobos):

                self.accuracyModelLog[i][epoch] = self.accuracy[i]
                self.lossModelLog[i][epoch] = self.loss[i]
                self.valAccuracyModelLog[i][epoch] = self.valAccuracy[i]
                self.valLossModelLog[i][epoch] = self.valLoss[i]
        
        self.setWeights(self.positionsLobos[0])

        with open('weedDetectionInWheat/CNN/MetaheuristicReport.txt', 'w') as f:

            for i in range(self.numeroLobos):

                f.write(','.join(map(str, self.accuracyModelLog[i])) + '\n') 
                f.write(','.join(map(str, self.lossModelLog[i])) + '\n') 
                f.write(','.join(map(str, self.valAccuracyModelLog[i])) + '\n') 
                f.write(','.join(map(str, self.valLossModelLog[i])) + '\n')

        return self.model
    
    def GWOExploracion(self, datasetEntrenamiento, datasetEvaluacion, iteracion):

        for n in range(self.numeroAgentes):

            print(f"Exploración Epoch {iteracion + 1} / {self.iterMaximo} (Agente {n + 1} / {self.numeroAgentes})| Entrenamiento | Validación: ")

            self.setWeights(self.positions[n])
            loss, accuracy = self.calcularPerdidaConPesos(datasetEntrenamiento, self.classWeight)
            valLoss, valAccuracy = self.model.evaluate(datasetEvaluacion, verbose=1)    

            for i in range(self.numeroLobos):

                if(loss < self.loss[i]):

                    print(f"Actualizacion {self.nombreLobos[i]}")
                    self.actualizarLobos(loss, accuracy, valLoss, valAccuracy, np.ravel(self.positions[n, :].copy()), i)
                    break
            
            for i in range(self.numeroLobos):

                print(f"{self.nombreLobos[i]} -> Perdida: {self.loss[i]}, Accuracy: {self.accuracy[i]}, valLoss: {self.valLoss[i]}, valAccuracy: {self.valAccuracy[i]}")

    def actualizarLobos(self, loss, accuracy, valLoss, valAccuracy, posiciones, lobo):

        for i in range(self.numeroLobos - 1, lobo - 1, -1):

            if(i == lobo):

                self.loss[i] = loss
                self.accuracy[i] = accuracy
                self.valLoss[i] = valLoss
                self.valAccuracy[i] = valAccuracy
                self.positionsLobos[i] = posiciones
            
            else:

                self.loss[i] = self.loss[i - 1]
                self.accuracy[i] = self.accuracy[i - 1]
                self.valLoss[i] = self.valLoss[i - 1]
                self.valAccuracy[i] = self.valAccuracy[i - 1] 
                self.positionsLobos[i] = self.positionsLobos[i - 1] 
        
    def GWOExplotacion(self, iteracion):

        for i in range(self.numeroAgentes):

            tiempoInicial = time.time()
            pesos = len(self.positions[0])

            a = 2 - iteracion * (2 / self.iterMaximo)

            posiciones = np.array(self.positions[i], dtype=np.float32)
            posicion = np.array(self.positionsLobos, dtype=np.float32)
            
            # Alojar en  GPU

            distanciaPosiciones = cuda.mem_alloc(posiciones.nbytes)
            cuda.memcpy_htod(distanciaPosiciones, posiciones)

            distanciaPosicionLobos= cuda.mem_alloc(posicion.nbytes)
            cuda.memcpy_htod(distanciaPosicionLobos, posicion)

            mod = SourceModule("""
            #define MAXLOBOS """ + str(self.numeroLobos) + """

            __device__ float xorshift32(unsigned int seed) {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                return (seed & 0x7FFFFFFF) / float(0x7FFFFFFF); // Normalizar a rango [0, 1]
            }
            __global__ void actualizar(float *posiciones, float *posicionesLobos, float a, int numeroPesos,
                                        float limiteInferior, float limiteSuperior,  unsigned int seed) {
                int thread = blockIdx.x * blockDim.x + threadIdx.x;
                
                if (thread < numeroPesos) {
                               
                    float A[MAXLOBOS];
                    float C[MAXLOBOS];
                    float posicionSiguiente[MAXLOBOS];
                    float solucion = 0.0;
                    unsigned int hilo_seed = seed + thread;
                               
                    for (int i = 0; i < MAXLOBOS; i++){

                        float r1 = xorshift32(hilo_seed + i);
                        float r2 = xorshift32(hilo_seed + i + MAXLOBOS);
                               
                        A[i] = 2 * a * r1 - a;
                        C[i] = 2 * a * r2 - a;
                    }   
                               
                    for (int i = 0; i < MAXLOBOS; i++){
                               
                        float posicionLoboActual = posicionesLobos[thread + (numeroPesos * i)];
                        float posicionPresa = posiciones[thread];

                        posicionSiguiente[i] = fabs(C[i] * posicionLoboActual - posicionPresa);
                        float X = posicionLoboActual - A[i] * posicionSiguiente[i];
                        solucion += X;
                    }
                               
                    solucion /= MAXLOBOS;
                               
                    posiciones[thread] = solucion;
                               
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
            seed = np.uint32(int(time.time() * 1000) % (2**32))

            actualizar_posiciones(distanciaPosiciones, distanciaPosicionLobos, np.float32(a), np.int32(pesos), np.float32(self.limiteInferior), 
                                np.float32(self.limiteSuperior), seed, block=(block, 1, 1), grid=(grid, 1))

            # Recuperamos los datos desde la GPU
            cuda.memcpy_dtoh(posiciones, distanciaPosiciones)
            self.positions[i] = posiciones

            tiempoFinal = time.time()
            print(f"Explotación {i + 1} / {self.numeroAgentes} tiempo de ejecución: {tiempoFinal - tiempoInicial} segundos")
import os
import time
import hashlib
import numpy as np
from tqdm import tqdm

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras.layers import Layer, Flatten, Dense, Input
from tensorflow.keras.models import load_model, Model

import pycuda.autoinit # type: ignore
import pycuda.driver as cuda  # type: ignore
from pycuda.compiler import SourceModule # type: ignore

from weedDetectionInWheat.metaheuristic.customLayers.maskLayer import MaskLayer

class GWO:
    def __init__(self, model, iterMaximo=10, numeroAgentes = 5,  numeroLobos = 5, classWeight = None, transferLearning = None, featureSelection = None):

        # Hiperparametros del constructor

        self.model = model
        self.iterMaximo = iterMaximo
        self.numeroAgentes = numeroAgentes
        self.numeroLobos = numeroLobos
        self.classWeight = classWeight
        self.transferLearning = transferLearning
        self.limiteSuperior = 0.5
        self.limiteInferior = -0.5

        self.features = None
        self.featureSelection = featureSelection
        self.numberFeatures = None
        self.weights_structure = None
        self.cantidadPesos = None

        self.nombreLobos = {0 : "Alfa", 1 : "Beta", 2: "Delta"}

        for i in range(self.numeroLobos - 3):

            self.nombreLobos[i + 3] = (f"Omega {i + 1}")

        if(featureSelection is None):

            self.weights_structure = model.get_weights()
            pesos = self.obtenerCantidadPesos()
            self.cantidadPesos = np.uint32(pesos)

        else:

            self.features = self.model.get_layer(featureSelection)
            self.inputFeatures = self.features.get_build_config()
            self.inputShape = self.inputFeatures["input_shape"]
            pesos = [dim for dim in self.inputShape if dim is not None]

            self.cantidadPesos = 1

            for valor in pesos:

                self.cantidadPesos *= valor

            self.cantidadPesos = np.uint32(self.cantidadPesos)

        self.positions = np.zeros((numeroAgentes, self.cantidadPesos))
        self.positionsNoRound = np.zeros((numeroAgentes, self.cantidadPesos))
        self.numberFeatures = np.zeros((numeroAgentes))
        self.positionsLobos = np.zeros((self.numeroLobos, self.cantidadPesos))
        self.numberFeaturesLobos = np.zeros((self.numeroLobos))

        self.loss = np.full(self.numeroLobos, np.finfo(np.float64).max)
        self.accuracy = np.zeros((self.numeroLobos))
        self.valLoss = np.full(self.numeroLobos, np.finfo(np.float64).max)
        self.valAccuracy = np.zeros((self.numeroLobos))
        
        self.lossModelLog = np.zeros((self.numeroLobos, self.iterMaximo))
        self.accuracyModelLog = np.zeros((self.numeroLobos, self.iterMaximo))
        self.valLossModelLog = np.zeros((self.numeroLobos, self.iterMaximo))
        self.valAccuracyModelLog = np.zeros((self.numeroLobos, self.iterMaximo))
        self.numberFeaturesLog = np.zeros((self.numeroLobos, self.iterMaximo))
        
        if(featureSelection is not None):

            for i in range(self.numeroAgentes):

                self.positions[i], self.positionsNoRound[i], self.numberFeatures[i] = self.asignarSelection()

            return

        elif(transferLearning is None):

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

    def setWeights(self, weights):

        new_weights = []
        idx = 0
        for w in self.weights_structure:
            
            shape = w.shape
            size = np.prod(shape)
            new_weights.append(np.array(weights[idx:idx + size]).reshape(w.shape)) # Asignar a la forma original
            idx += size

        self.model.set_weights(new_weights)

    def asignarPosicion(self): # Generar una matriz con todos los pesos a optimizar de la red.

        pocisionRandom = []

        for w in self.weights_structure:

            # Generar una matriz de valores aleatorios con la misma forma que los pesos 'w'
            random_weights = np.random.uniform(self.limiteInferior, self.limiteSuperior, w.shape)
            pocisionRandom.append(random_weights.flatten())

        return np.concatenate(pocisionRandom)

    def asignarSelection(self):

        pocision = []
        posicionNoRound = []
        cantidadFeatures = 0

        for i in range(self.cantidadPesos):

            random_weights = np.random.uniform(self.limiteInferior, self.limiteSuperior)
            posicionNoRound.append(random_weights)

        for i in range(self.cantidadPesos):

            sigmoid = 1 / (1 + np.exp(posicionNoRound[i]))

            if(sigmoid > 0.5):
                cantidadFeatures += 1
                pocision.append(1)
            else:
                pocision.append(0)

        return np.array(pocision), np.array(posicionNoRound), cantidadFeatures

    def modificarModelo(self, mascara):

        capasEntrada = self.model.get_layer("conv2d").input
        capasSalida = self.model.get_layer('flatten').output

        mascaraLayer = MaskLayer(mask=mascara)(capasSalida)

        capasSalida = mascaraLayer
        agregarLayers = False 

        for layer in self.model.layers: # Agregar las capas restantes personalizadas
            if layer.name == 'dense':
                agregarLayers = True
            if agregarLayers:
                capasSalida = layer(capasSalida)

        modeloCustom = Model(inputs=capasEntrada, outputs=capasSalida)

        modeloCustom.compile(
            loss='binary_crossentropy',
            optimizer=tf.keras.optimizers.Adam(0.001),
            metrics=['accuracy'],
        )

        self.model = modeloCustom

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
                return (seed & 0x7FFFFFFF) / float(0x7FFFFFFF) * 2;
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

    def calcularPerdidaFeatures(self, dataset, classWeight, iteracion):

        loss = 0.0
        accuracy = 0.0
        total = 0

        alfa = 0.99
        beta = 0.01
        
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
        
        loss = alfa * loss + beta * (self.numberFeatures[iteracion] / self.cantidadPesos)

        print(f"Loss: {loss/total}, Accuracy: {accuracy / total}, Features = {self.numberFeatures[iteracion]}")
        return (loss / total), (accuracy / total), self.numberFeatures[iteracion]

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
        self.escribirReporte()
        return self.model
    
    def optimizeFeature(self, datasetEntrenamiento, datasetEvaluacion):

        for epoch in range(self.iterMaximo):

            self.GWOExploracionFeatures(datasetEntrenamiento, datasetEvaluacion, epoch)
            self.GWOExplotacionFeatures(epoch)

            for i in range(self.numeroLobos):

                self.accuracyModelLog[i][epoch] = self.accuracy[i]
                self.lossModelLog[i][epoch] = self.loss[i]
                self.valAccuracyModelLog[i][epoch] = self.valAccuracy[i]
                self.valLossModelLog[i][epoch] = self.valLoss[i] 
                self.numberFeaturesLog[i][epoch] = self.numberFeaturesLobos[i]           

        for i in range(len(self.positionsLobos[0])):

            if(self.positionsLobos[0, i] > 0.5):
                self.positionsLobos[0, i] = 1
            else:
                self.positionsLobos[0, i] = 0

        self.modificarModelo(self.positionsLobos[0])
        self.escribirReporte()
        return self.model
    
    def GWOExploracion(self, datasetEntrenamiento, datasetEvaluacion, iteracion):

        for n in range(self.numeroAgentes):

            print(f"Exploración Epoch {iteracion + 1} / {self.iterMaximo} (Agente {n + 1} / {self.numeroAgentes})| Entrenamiento | Validación: ")

            self.setWeights(self.positions[n])
            loss, accuracy = self.calcularPerdidaConPesos(datasetEntrenamiento, self.classWeight, n)
            valLoss, valAccuracy = self.model.evaluate(datasetEvaluacion, verbose=1)    

            for i in range(self.numeroLobos):

                if(loss < self.loss[i]):

                    print(f"Actualizacion {self.nombreLobos[i]}")
                    self.actualizarLobos(loss, accuracy, valLoss, valAccuracy, np.ravel(self.positions[n, :].copy()), i)
                    break
            
            for i in range(self.numeroLobos):

                print(f"{self.nombreLobos[i]} -> Perdida: {self.loss[i]}, Accuracy: {self.accuracy[i]}, valLoss: {self.valLoss[i]}, valAccuracy: {self.valAccuracy[i]}")
    
    def GWOExploracionFeatures(self, datasetEntrenamiento, datasetEvaluacion, iteracion):

        for n in range(self.numeroAgentes):

            print(f"Exploración Epoch {iteracion + 1} / {self.iterMaximo} (Agente {n + 1} / {self.numeroAgentes})| Entrenamiento | Validación: ")

            self.modificarModelo(self.positions[n])
        
            loss, accuracy, numberFeatures = self.calcularPerdidaFeatures(datasetEntrenamiento, self.classWeight, n)
            valLoss, valAccuracy = self.model.evaluate(datasetEvaluacion, verbose=1)    

            for i in range(self.numeroLobos):

                if(loss < self.loss[i]):

                    print(f"Actualizacion {self.nombreLobos[i]}")
                    self.actualizarLobos(loss, accuracy, valLoss, valAccuracy, numberFeatures, np.ravel(self.positionsNoRound[n, :].copy()), i)
                    break
            
            for i in range(self.numeroLobos):

                print(f"{self.nombreLobos[i]} -> Perdida: {self.loss[i]}, Accuracy: {self.accuracy[i]}, valLoss: {self.valLoss[i]}, valAccuracy: {self.valAccuracy[i]}, Features: {self.numberFeaturesLobos[i]}")
        
    def actualizarLobos(self, loss, accuracy, valLoss, valAccuracy, numberFeatures, posiciones, lobo):

        for i in range(self.numeroLobos - 1, lobo - 1, -1):

            if(i == lobo):

                self.loss[i] = loss
                self.accuracy[i] = accuracy
                self.valLoss[i] = valLoss
                self.valAccuracy[i] = valAccuracy
                self.numberFeaturesLobos[i] = numberFeatures
                self.positionsLobos[i] = posiciones
            
            else:

                self.loss[i] = self.loss[i - 1]
                self.accuracy[i] = self.accuracy[i - 1]
                self.valLoss[i] = self.valLoss[i - 1]
                self.valAccuracy[i] = self.valAccuracy[i - 1] 
                self.numberFeaturesLobos[i] = self.numberFeaturesLobos[i - 1]
                self.positionsLobos[i] = self.positionsLobos[i - 1]     

    def escribirReporte(self):

        with open('weedDetectionInWheat/CNN/MetaheuristicReport.txt', 'w') as f:

            for i in range(self.numeroLobos):

                f.write(','.join(map(str, self.accuracyModelLog[i])) + '\n') 
                f.write(','.join(map(str, self.lossModelLog[i])) + '\n') 
                f.write(','.join(map(str, self.valAccuracyModelLog[i])) + '\n') 
                f.write(','.join(map(str, self.valLossModelLog[i])) + '\n')

                if(self.featureSelection is not None):

                    f.write(','.join(map(str, self.numberFeaturesLog[i])) + '\n')

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

    def GWOExplotacionFeatures(self, iteracion):

            for i in range(self.numeroAgentes):

                tiempoInicial = time.time()
                pesos = len(self.positions[0])

                a = 2 - iteracion * (2 / self.iterMaximo)

                posiciones = np.array(self.positions[i], dtype=np.float32)
                perdida = np.array(self.loss, dtype=np.float32)
                posicionesNoRound = np.array(self.positions[i], dtype=np.float32)
                posicion = np.array(self.positionsLobos, dtype=np.float32)

                perdidaTotal = sum(self.loss)

                for i in range(self.numeroLobos):
                    perdida[i] = self.loss[i] / perdidaTotal
                
                # Alojar en  GPU

                distanciaPosiciones = cuda.mem_alloc(posiciones.nbytes)
                cuda.memcpy_htod(distanciaPosiciones, posiciones)

                perdidaNormalizada = cuda.mem_alloc(self.loss.nbytes)
                cuda.memcpy_htod(perdidaNormalizada, perdida)

                distanciaPosicionesNoRound = cuda.mem_alloc(posicionesNoRound.nbytes)
                cuda.memcpy_htod(distanciaPosicionesNoRound, posicionesNoRound)

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
                __global__ void actualizar(float *posiciones, float *loss, float *posicionesNoRound, float *posicionesLobos, float a, int numeroPesos, float limiteInferior, float limiteSuperior, unsigned int seed) {

                    int thread = blockIdx.x * blockDim.x + threadIdx.x;
                    
                    if (thread < numeroPesos) {
                                
                        float A[MAXLOBOS];
                        float C[MAXLOBOS];
                        float posicionSiguiente[MAXLOBOS];
                        float solucion = 0.0;
                        unsigned int hilo_seed = seed + thread;
                                
                        for (int i = 0; i < MAXLOBOS; i++){

                            unsigned int seed1 = hilo_seed ^ (i * 2654435761U) ^ (hilo_seed >> 13);
                            unsigned int seed2 = hilo_seed ^ ((i + MAXLOBOS) * 2246822519U) ^ (hilo_seed << 7);
                        
                            float r1 = xorshift32(seed1);
                            float r2 = xorshift32(seed2);
                                
                            A[i] = 2 * a * r1 - a;
                            C[i] = 2 * a * r2 - a;
                        }   
                                
                        for (int i = 0; i < MAXLOBOS; i++){
                                
                            float posicionLoboActual = posicionesLobos[thread + (numeroPesos * i)];
                            float posicionPresa = posiciones[thread];

                            posicionSiguiente[i] = fabs(C[i] * posicionLoboActual - posicionPresa);
                            float X = posicionLoboActual - A[i] * posicionSiguiente[i];
                            X *= loss[i];
                            solucion += X;
                        }
                                
                        solucion /= MAXLOBOS;
                        posiciones[thread] = 1 / (1 + exp(-solucion));
                        posicionesNoRound[thread] = posiciones[thread];
        
                        if(posiciones[thread] < limiteInferior){

                            posicionesNoRound[thread] = limiteInferior;            
                        }
                        else if(posiciones[thread] > limiteSuperior){
                                    
                            posicionesNoRound[thread] = limiteSuperior;
                        }    

                        if (posiciones[thread] < 0.5) {
                            posiciones[thread] = 0;
                        } 
                        else {
                            posiciones[thread] = 1;
                        }                     
                    }
                }
                """)

                # Inicializar y ejecutar el kernel
                actualizar_posiciones = mod.get_function("actualizar")
                block = 1024
                grid = (pesos + block - 1) // block
                seed = self.generarSemilla()

                actualizar_posiciones(distanciaPosiciones, perdidaNormalizada, distanciaPosicionesNoRound, distanciaPosicionLobos, np.float32(a), np.int32(pesos), 
                                      np.float32(self.limiteInferior),  np.float32(self.limiteSuperior), seed, block=(block, 1, 1), grid=(grid, 1))

                # Recuperamos los datos desde la GPU

                cuda.memcpy_dtoh(posiciones, distanciaPosiciones)
                cuda.memcpy_dtoh(posicionesNoRound, distanciaPosicionesNoRound)

                self.positions[i] = posiciones
                self.positionsNoRound[i] = posicionesNoRound

                numeroFeatures = 0

                for feature in self.positions[i]:
                    if(feature == 1):
                        numeroFeatures += 1

                self.numberFeatures[i] = numeroFeatures

                tiempoFinal = time.time()
                print(f"Explotación {i + 1} / {self.numeroAgentes} tiempo de ejecución: {tiempoFinal - tiempoInicial} segundos")

    def generarSemilla(self):
        # Estrategia 1: Entropía del sistema operativo con os.urandom
        entropia = int.from_bytes(os.urandom(4), byteorder='big')
        
        # Estrategia 2: Tiempo en nanosegundos combinado con hash
        tiempoActual = time.time_ns()
        tiempoHash = int(hashlib.sha256(str(tiempoActual).encode()).hexdigest(), 16)
        
        # Estrategia 3: Generador aleatorio de NumPy sin depender del tiempo
        numeroAleatorio = np.random.randint(0, 2**32, dtype=np.uint32)
        
        # Combinar las tres fuentes
        semilla = (entropia ^ tiempoHash ^ numeroAleatorio) % (2**32)
        return np.uint32(semilla)
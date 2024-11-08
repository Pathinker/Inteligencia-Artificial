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

        prediccionesCorrectas = 0
        total = 0
        loss = 0.0
        
        for x, y in tqdm(dataset, desc = f"Calculando Perdida", unit="batch"):

            # Realizar una predicción por batch y extraer su perdida.

            yPrediccion = self.model.predict(x, verbose = 0) 
            lossBatch = tf.keras.losses.binary_crossentropy(y, yPrediccion)

            if classWeight is not None:
                             
                weights = np.zeros_like(lossBatch.numpy())  # Crear un arreglo de ceros con la misma forma que loss_batch
                etiqueta = y.numpy().flatten() 

                for i in range(len(etiqueta)): 
                    label = etiqueta[i]
                    weights[i] = classWeight[label]    

                lossBatch *=  weights
            
            loss += tf.reduce_sum(lossBatch).numpy()
            total += len(y)

            prediccionClase = tf.round(yPrediccion)
            prediccionesCorrectas += tf.reduce_sum(tf.cast(tf.equal(prediccionClase, y), tf.float32)).numpy()  # Contar aciertos transformando un tensor ft float 32.
        
        accuracy = prediccionesCorrectas / total
        print(f"Precisión: {accuracy} Pérdida: {loss / total}")

        return loss / total  
    
    def optimize(self, datasetEntrenamiento, datasetEvaluacion):

        for iteracion in range(self.iterMaximo):

            self.GWOExploracion(datasetEntrenamiento, datasetEvaluacion, iteracion)
            self.GWOExplotacion(iteracion)
        
        self.setWeights(self.posicionAlfa)
    
    def GWOExploracion(self, datasetEntrenamiento, datasetEvaluacion, iteracion):

        for n in range(self.numeroAgentes):

            print(f"Exploración Epoch {iteracion + 1} / {self.iterMaximo} (Agente {n + 1} / {self.numeroAgentes})| Entrenamiento | Validación: ")

            self.setWeights(self.positions[n])
            loss = self.calcularPerdidaConPesos(datasetEntrenamiento, self.classWeight)
            #loss, _ = self.model.evaluate(datasetEntrenamiento, verbose = 1)
            self.model.evaluate(datasetEvaluacion, verbose=1)    

            # Actualizar alpha, beta y delta al detectar un agente menor equivalente a una menor perdida

            print("Error: ", loss)

            if loss < self.lossAlfa:

                print("Actualización Alfa: ", self.lossAlfa)
                print("Posiciones Alfa: ", self.posicionAlfa)

                # Actualizar Alpha y mover los otros lobos hacia abajo
                
                self.lossDelta, self.posicionDelta = self.lossBeta, np.ravel(self.posicionBeta.copy())
                self.lossBeta, self.posicionBeta = self.lossAlfa, np.ravel(self.posicionAlfa.copy())
                self.lossAlfa, self.posicionAlfa = loss, np.ravel(self.positions[n, :].copy())

            elif loss < self.lossBeta:

                print("Actualización Beta: ", self.lossBeta)
                print("Posiciones Beta: ", self.posicionBeta)

                self.lossDelta, self.posicionDelta = self.lossBeta, np.ravel(self.posicionBeta.copy())
                self.lossBeta, self.posicionBeta = loss, np.ravel(self.positions[n, :].copy())

            elif loss < self.lossDelta:

                print("Actualización Delta: ", self.lossDelta)
                print("Posiciones Delta: ", self.posicionDelta)
                    
                self.lossDelta, self.posicionDelta = loss, np.ravel(self.positions[n, :].copy())

            print("Perdida Alfa: ", self.lossAlfa)
            print("Posiciones Alfa: ", self.posicionAlfa)
            print("Perdida Beta: ", self.lossBeta)
            print("Posiciones Beta: ", self.posicionBeta)
            print("Perdida Delta: ", self.lossDelta)
            print("Posiciones Delta: ", self.posicionDelta)

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

            print("Pesos Previo GWO: ", self.positions[i])
                
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

            print("Pesos Después GWO: ", self.positions[i])
            print("Peso Maximo:", np.max(self.positions[i]))
            print("Peso Minimo", np.min(self.positions[i]))

            tiempoFinal = time.time()
            print(f"Explotación {i + 1} / {self.numeroAgentes} tiempo de ejecución: {tiempoFinal - tiempoInicial} segundos")
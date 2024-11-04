import numpy as np
import tensorflow as tf
from tqdm import tqdm

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
    
    def optimize(self, datasetEntrenamiento, datasetEvaluacion):

        for iteracion in range(self.iterMaximo):

            #self.GWOExploracion(datasetEntrenamiento, datasetEvaluacion, iteracion)
            self.GWOExplotacion(iteracion)
        
        return self.posicionAlfa
    
    def GWOExploracion(self, datasetEntrenamiento, datasetEvaluacion, iteracion):

        for n in range(self.numeroAgentes):

            print(f"Exploración Epoch {iteracion + 1} / {self.iterMaximo} (Agente {n + 1} / {self.numeroAgentes})| Entrenamiento | Validación: ")

            self.setWeights(self.positions[n])
            loss = self.calcularPerdidaConPesos(datasetEntrenamiento, self.classWeight)
            self.loss.append(loss)

            # Evaluar la pérdida en los datos de evaluación

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

        a = 2 - iteracion * (2 / self.iterMaximo)
        pesos = len(self.positions[0])
        agentes = self.numeroAgentes

        posiciones = np.array(self.positions, dtype=np.float32)
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
        
        mod = SourceModule("""
        #include <curand_kernel.h>
                        
        __global__ void actualizar(float *posiciones, float *posicionAlfa, float *posicionBeta, float *posicionDelta, float a, int num_pesos, int num_agentes, unsigned long long seed) {
            int tid = blockIdx.x * blockDim.x + threadIdx.x;
            
            if (tid < num_agentes * num_pesos) {
                curandState state;
                curand_init(seed, tid, 0, &state);
                
                int n = tid / num_pesos;
                int i = tid % num_pesos;

                float r1 = curand_uniform(&state);
                float r2 = curand_uniform(&state);
                float A1 = 2 * a * r1 - a;
                float C1 = 2 * r2;

                r1 = curand_uniform(&state);
                r2 = curand_uniform(&state);
                float A2 = 2 * a * r1 - a;
                float C2 = 2 * r2;

                r1 = curand_uniform(&state);
                r2 = curand_uniform(&state);
                float A3 = 2 * a * r1 - a;
                float C3 = 2 * r2;

                float posicionAlfa_i = posicionAlfa[i];
                float posicionBeta_i = posicionBeta[i];
                float posicionDelta_i = posicionDelta[i];
                float posicionSolucion_i = posiciones[n * num_pesos + i];

                float distanciaAlfa = fabs(C1 * posicionAlfa_i - posicionSolucion_i);
                float distanciaBeta = fabs(C2 * posicionBeta_i - posicionSolucion_i);
                float distanciaDelta = fabs(C3 * posicionDelta_i - posicionSolucion_i);

                float X1 = posicionAlfa_i - A1 * distanciaAlfa;
                float X2 = posicionBeta_i - A2 * distanciaBeta;
                float X3 = posicionDelta_i - A3 * distanciaDelta;

                posiciones[n * num_pesos + i] = (X1 + X2 + X3) / 3;
            }
        }
        """, no_extern_c=True)

        # Inicializar y ejecutar el kernel
        actualizar_posiciones = mod.get_function("?__device_stub__Z10actualizarPfS_S_S_fiiy@@YAXPEAM000MHH_K@Z")
        block = 1024
        grid = (pesos * agentes + block - 1) // block

        actualizar_posiciones(distanciaPosiciones, distanciaPosicionAlfa, distanciaPosicionBeta, distanciaPoscionDelta, 
                            np.float32(a), np.int32(pesos), np.int32(agentes), block=(block, 1, 1), grid=(grid, 1))

        # Recuperamos los datos desde la GPU
        cuda.memcpy_dtoh(posiciones, distanciaPosiciones)
        self.positions = posiciones.reshape(agentes, pesos)
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

from sklearn.svm import SVC # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
from sklearn.pipeline import Pipeline # type: ignore
from sklearn.metrics import accuracy_score # type: ignore

from weedDetectionInWheat.metaheuristic.customLayers.maskLayer import MaskLayer

class GWO:
    def __init__(self, model, epochs=10, agents = 5,  wolves = 5, class_weight = None, transfer_learning = None, feature_selection = None, ensemble_model = None):

        self.model = model
        self.epochs = epochs
        self.agents = agents
        self.wolves = wolves
        self.wolves_name = {0 : "Alfa", 1 : "Beta", 2: "Delta"}
        self.class_weight = class_weight
        self.transfer_learning = transfer_learning
        self.feature_selection = feature_selection
        self.ensemble_model = ensemble_model
        self.upper_bound = 0.5
        self.lower_bound = -0.5

        self.features = None
        self.number_features = None
        self.weights_structure = None
        self.number_weights = None

        for wolf in range(self.wolves - 3):

            self.wolves_name[wolf + 3] = (f"Omega {wolf + 1}")

        if(feature_selection is None):

            self.weights_structure = model.get_weights()
            weights = self.get_number_weights()
            self.number_weights = np.uint32(weights)

        else:

            self.features = self.model.get_layer(feature_selection)
            self.input_features = self.features.get_build_config()
            self.input_shape = self.input_features["input_shape"]
            weights = [dim for dim in self.input_shape if dim is not None]

            self.number_weights = 1

            for value in weights:

                self.number_weights *= value

            self.number_weights = np.uint32(self.number_weights)

        self.positions = np.zeros((agents, self.number_weights))
        self.round_positions = np.zeros((agents, self.number_weights))
        self.number_features = np.zeros((agents))
        self.wolves_positions = np.zeros((self.wolves, self.number_weights))
        self.number_features_wolves = np.zeros((self.wolves))

        self.loss = np.full(self.wolves, np.finfo(np.float64).max)
        self.accuracy = np.zeros((self.wolves))
        self.validation_loss = np.full(self.wolves, np.finfo(np.float64).max)
        self.validation_accuracy = np.zeros((self.wolves))
        
        self.loss_log = np.zeros((self.wolves, self.epochs))
        self.accuracy_log = np.zeros((self.wolves, self.epochs))
        self.validation_loss_log = np.zeros((self.wolves, self.epochs))
        self.validation_accuracy_log = np.zeros((self.wolves, self.epochs))
        self.number_features_log = np.zeros((self.wolves, self.epochs))
        
        if(feature_selection is not None):

            for i in range(self.agents):

                self.round_positions[i], self.positions[i], self.number_features[i] = self.set_selection()

            return

        elif(transfer_learning is None):

            for i in range(self.agents):

                self.round_positions[i] = self.set_position()

        else:

            self.set_learning()

        self.set_weights(self.round_positions[0])

    def get_number_weights(self):
        
        total_weights = 0

        for weights in self.weights_structure:

            elementos = np.prod(weights.shape)
            total_weights += elementos

        return total_weights

    def set_weights(self, weights):

        new_weights = []
        index = 0

        for weights in self.weights_structure:
            
            shape = weights.shape
            size = np.prod(shape)
            new_weights.append(np.array(weights[index:index + size]).reshape(weights.shape))
            index += size

        self.model.set_weights(new_weights)

    def set_position(self):

        random_position = []

        for w in self.weights_structure:

            random_weights = np.random.uniform(self.lower_bound, self.upper_bound, w.shape)
            random_position.append(random_weights.flatten())

        return np.concatenate(random_position)

    def set_selection(self):

        position = []
        round_position = []
        cantidadFeatures = 0

        for i in range(self.number_weights):

            random_weights = np.random.uniform(self.lower_bound, self.upper_bound)
            position.append(random_weights)

        for i in range(self.number_weights):

            sigmoid = 1 / (1 + np.exp(position[i]))

            if(sigmoid > 0.5):
                cantidadFeatures += 1
                round_position.append(1)
            else:
                round_position.append(0)

        return np.array(round_position), np.array(position), cantidadFeatures

    def set_mask(self, mask):

        entry_layer = self.model.get_layer("conv2d").input
        output_layer = self.model.get_layer('flatten').output

        mask_layer = MaskLayer(mask=mask)(output_layer)

        output_layer = mask_layer
        feature_selection_layers = False 

        for layer in self.model.layers:
            if layer.name == 'dense':
                feature_selection_layers = True
            if feature_selection_layers:
                output_layer = layer(output_layer)

        custom_model = Model(inputs=entry_layer, outputs=output_layer)

        custom_model.compile(
            loss='binary_crossentropy',
            optimizer=tf.keras.optimizers.Adam(0.001),
            metrics=['accuracy'],
        )

        self.model = custom_model

    def set_learning(self):

        self.transfer_learning= self.model.get_weights()
        flattened_weights = np.concatenate([weight.flatten() for weight in self.transfer_learning])
        self.transfer_learning = flattened_weights

        for i in range(self.agents):

            inicial_time = time.time()
            weights = len(self.transfer_learning)

            positions = np.array(self.round_positions[i], dtype=np.float32)
            transfer_learning = np.array(self.transfer_learning, dtype=np.float32)

            positions_distance = cuda.mem_alloc(positions.nbytes)
            cuda.memcpy_htod(positions_distance, positions)

            transfer_learning = cuda.mem_alloc(self.transfer_learning.nbytes)
            cuda.memcpy_htod(transfer_learning, self.transfer_learning)

            mod = SourceModule("""

            __device__ float xorshift32(unsigned int seed) {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                return (seed & 0x7FFFFFFF) / float(0x7FFFFFFF) * 2;
            }
            __global__ void update(float *positions, float *transfer_learning, int weights_number,
                                        float lower_bound, float upper_bound,  unsigned int seed) {
                int thread = blockIdx.x * blockDim.x + threadIdx.x;
                
                if (thread < weights_number) {
                               
                    unsigned int thread_seed = seed + thread;
                    float random = xorshift32(thread_seed);
            
                    positions[thread] = transfer_learning[thread] * random;
                               
                    if(positions[thread] < lower_bound){

                        positions[thread] = lower_bound;            
                    }
                    else if(positions[thread] > upper_bound){
                               
                        positions[thread] = upper_bound;
                    }               
                }
            }
            """)

            update_positions = mod.get_function("update")
            block = 1024
            grid = (weights + block - 1) // block
            seed = np.uint32(int(time.time() * 1000) % (2**32))

            update_positions(positions_distance, transfer_learning, np.int32(weights), np.float32(self.lower_bound), 
                                np.float32(self.upper_bound), seed, block=(block, 1, 1), grid=(grid, 1))

            cuda.memcpy_dtoh(positions, positions_distance)
            self.round_positions[i] = positions

            final_time = time.time()
            print(f"Inicialización {i + 1} / {self.agents} tiempo de ejecución: {final_time - inicial_time} segundos")

    def weighted_loss(self, dataset, class_weight):

        loss = 0.0
        accuracy = 0.0
        total = 0
        
        for x, y in tqdm(dataset, desc = f"Calculando Perdida", unit="batch"):

            weighted_losses = []

            for label in y:

                weight = class_weight[int(label)]
                weighted_losses.append(weight)

            weighted_losses = np.array(weighted_losses)
            
            loss_batch, accuracy_batch = self.model.evaluate(x, y, sample_weight = weighted_losses, verbose = 0)

            loss += loss_batch
            accuracy += accuracy_batch
            total += 1

        print(f"loss: {loss/total}, accuracy = {accuracy / total}")
        return (loss / total), (accuracy / total)

    def loss_features(self, dataset, class_weight, epoch):

        loss = 0.0
        accuracy = 0.0
        total = 0

        alfa = 0.99
        beta = 0.01
        
        for x, y in tqdm(dataset, desc = f"Calculando Perdida", unit="batch"):

            weighted_losses = []

            for i in y:

                peso = class_weight[int(i)]
                weighted_losses.append(peso)

            weighted_losses = np.array(weighted_losses)
            
            loss_batch, accuracy_batch = self.model.evaluate(x, y, sample_weight = weighted_losses, verbose = 0)

            loss += loss_batch
            accuracy += accuracy_batch
            total += 1
        
        loss = alfa * loss + beta * (self.number_features[epoch] / self.number_weights)

        print(f"Loss: {loss/total}, Accuracy: {accuracy / total}, Features = {self.number_features[epoch]}")
        return (loss / total), (accuracy / total), self.number_features[epoch]

    def loss_ensemble(self, trainDataset, evaluationDataset, class_weight, epoch):

        inicial_time = time.time()

        alfa = 0.99
        beta = 0.01

        layer_name = "conv2d"
        entry_layer = self.model.get_layer(layer_name)

        layer_name = "mask"  
        output_layer = self.model.get_layer(layer_name)

        flatten_alexnet = Model(inputs = entry_layer.input, outputs = output_layer.output)

        svm = Pipeline([

            ("scaler", StandardScaler()),
            ("svm", SVC(C = 1, kernel = "rbf", gamma = "scale", verbose = True))

        ])

        def get_convolution(dataset, model):
  
            features = []
            labels = []

            for images, batch_labels in dataset:

                # Extraer características de la capa flatten de cada una de las imágenes
                batchFeatures = model(images, training=False)
                features.append(batchFeatures.numpy())  # Convertir a numpy
                labels.append(batch_labels.numpy())  # Obtener las etiquetas
            
            x = np.concatenate(features)
            y = np.concatenate(labels)
            y = y.ravel()

            return x, y

        x_train, y_train = get_convolution(trainDataset, flatten_alexnet)
        x_validation, y_validation = get_convolution(evaluationDataset, flatten_alexnet)        
        svm.fit(x_train, y_train)

        def get_loss(x, y, svm, class_weight):

            predict = svm.predict(x)
            decision_function = svm.decision_function(x)
            probabilities = 1 / (1 + np.exp(-decision_function)) 

            binary_crossentropy = tf.keras.losses.binary_crossentropy(y, probabilities)
            losses_list = []

            for label in y:

                loss = class_weight[label]
                losses_list.append(loss)

            losses_list = np.array(losses_list)
            weighted_loss = binary_crossentropy * losses_list
            weighted_loss = np.mean(weighted_loss)
            
            correct_predictions = []

            for estimate, expected in zip(predict, y):

                if(estimate > 0.5):
                    estimate = 1
                else:
                    estimate = 0

                if(estimate == expected):
                    correct_predictions.append(1)
                else:
                    correct_predictions.append(0)
            
            accuracy = sum(correct_predictions) / len(correct_predictions)
            weighted_loss = alfa * weighted_loss + beta * (self.number_features[epoch] / self.number_weights)
            
            return weighted_loss, accuracy
        
        train_loss, train_accuracy = get_loss(x_train, y_train, svm, class_weight)
        validation_loss, validation_accuracy = get_loss(x_validation, y_validation, svm, class_weight)

        final_time = time.time()

        print(f"Execution Time: {final_time - inicial_time} seconds")
        print(f"Loss: {train_loss}, Accuracy: {train_accuracy}, Validation Loss: {validation_loss}, Validation Accuracy: {validation_accuracy}, Features = {self.number_features[epoch]}")
        return train_loss, train_accuracy, validation_loss, validation_accuracy,  self.number_features[epoch] 
                    
    def optimize(self, datasetEntrenamiento, datasetEvaluacion):

        for epoch in range(self.epochs):

            self.GWO_exploration(datasetEntrenamiento, datasetEvaluacion, epoch)
            self.GWO_explotation(epoch)

            for i in range(self.wolves):

                self.accuracy_log[i][epoch] = self.accuracy[i]
                self.loss_log[i][epoch] = self.loss[i]
                self.validation_accuracy_log[i][epoch] = self.validation_accuracy[i]
                self.validation_loss_log[i][epoch] = self.validation_loss[i]
        
        self.set_weights(self.wolves_positions[0])
        self.get_report()
        return self.model
    
    def optimize_feature(self, datasetEntrenamiento, datasetEvaluacion):

        for epoch in range(self.epochs):

            self.GWO_feature_exploration(datasetEntrenamiento, datasetEvaluacion, epoch)
            self.GWO_feature_explotation(epoch)

            for i in range(self.wolves):

                self.accuracy_log[i][epoch] = self.accuracy[i]
                self.loss_log[i][epoch] = self.loss[i]
                self.validation_accuracy_log[i][epoch] = self.validation_accuracy[i]
                self.validation_loss_log[i][epoch] = self.validation_loss[i] 
                self.number_features_log[i][epoch] = self.number_features_wolves[i]           

        for i in range(len(self.wolves_positions[0])):

            if(self.wolves_positions[0, i] > 0.5):
                self.wolves_positions[0, i] = 1
            else:
                self.wolves_positions[0, i] = 0

        self.set_mask(self.wolves_positions[0])
        self.get_report()
        return self.model
    
    def GWO_exploration(self, datasetEntrenamiento, datasetEvaluacion, epoch):

        for n in range(self.agents):

            print(f"Exploración Epoch {epoch + 1} / {self.epochs} (Agente {n + 1} / {self.agents})| Entrenamiento | Validación: ")

            self.set_weights(self.round_positions[n])
            loss, accuracy = self.weighted_loss(datasetEntrenamiento, self.class_weight, n)
            validation_loss, validation_accuracy = self.model.evaluate(datasetEvaluacion, verbose=1)    

            for i in range(self.wolves):

                if(loss < self.loss[i]):

                    print(f"Actualizacion {self.wolves_name[i]}")
                    self.update_wolves(loss, accuracy, validation_loss, validation_accuracy, np.ravel(self.round_positions[n, :].copy()), i)
                    break
            
            for i in range(self.wolves):

                print(f"{self.wolves_name[i]} -> Perdida: {self.loss[i]}, Accuracy: {self.accuracy[i]}, validation_loss: {self.validation_loss[i]}, validation_accuracy: {self.validation_accuracy[i]}")
    
    def GWO_feature_exploration(self, datasetEntrenamiento, datasetEvaluacion, epoch):

        for n in range(self.agents):

            print(f"Exploración Epoch {epoch + 1} / {self.epochs} (Agente {n + 1} / {self.agents})| Entrenamiento | Validación: ")

            self.set_mask(self.round_positions[n])

            if(self.ensemble_model is None):
                loss, accuracy, number_features = self.loss_features(datasetEntrenamiento, self.class_weight, n)
                validation_loss, validation_accuracy = self.model.evaluate(datasetEvaluacion, verbose=1)    
            else:
                loss, accuracy, validation_loss, validation_accuracy, number_features = self.loss_ensemble(datasetEntrenamiento, datasetEvaluacion, self.class_weight, n)

            for i in range(self.wolves):

                if(loss < self.loss[i]):

                    print(f"Actualizacion {self.wolves_name[i]}")
                    self.update_wolves(loss, accuracy, validation_loss, validation_accuracy, number_features, np.ravel(self.positions[n, :].copy()), i)
                    break
            
            for i in range(self.wolves):

                print(f"{self.wolves_name[i]} -> Perdida: {self.loss[i]}, Accuracy: {self.accuracy[i]}, validation_loss: {self.validation_loss[i]}, validation_accuracy: {self.validation_accuracy[i]}, Features: {self.number_features_wolves[i]}")
        
    def update_wolves(self, loss, accuracy, validation_loss, validation_accuracy, number_features, positions, wolf):

        for i in range(self.wolves - 1, wolf - 1, -1):

            if(i == wolf):

                self.loss[i] = loss
                self.accuracy[i] = accuracy
                self.validation_loss[i] = validation_loss
                self.validation_accuracy[i] = validation_accuracy
                self.number_features_wolves[i] = number_features
                self.wolves_positions[i] = positions
            
            else:

                self.loss[i] = self.loss[i - 1]
                self.accuracy[i] = self.accuracy[i - 1]
                self.validation_loss[i] = self.validation_loss[i - 1]
                self.validation_accuracy[i] = self.validation_accuracy[i - 1] 
                self.number_features_wolves[i] = self.number_features_wolves[i - 1]
                self.wolves_positions[i] = self.wolves_positions[i - 1]     

    def get_report(self):

        with open('weedDetectionInWheat/CNN/MetaheuristicReport.txt', 'w') as f:

            for i in range(self.wolves):

                f.write(','.join(map(str, self.accuracy_log[i])) + '\n') 
                f.write(','.join(map(str, self.loss_log[i])) + '\n') 
                f.write(','.join(map(str, self.validation_accuracy_log[i])) + '\n') 
                f.write(','.join(map(str, self.validation_loss_log[i])) + '\n')

                if(self.feature_selection is not None):

                    f.write(','.join(map(str, self.number_features_log[i])) + '\n')

    def GWO_explotation(self, epoch):

        for i in range(self.agents):

            inicial_time = time.time()
            weights = len(self.round_positions[0])

            a = 2 - epoch * (2 / self.epochs)

            positions = np.array(self.round_positions[i], dtype=np.float32)
            posicion = np.array(self.wolves_positions, dtype=np.float32)
            
            # Alojar en  GPU

            positions_distance = cuda.mem_alloc(positions.nbytes)
            cuda.memcpy_htod(positions_distance, positions)

            positions_distance_wolves= cuda.mem_alloc(posicion.nbytes)
            cuda.memcpy_htod(positions_distance_wolves, posicion)

            mod = SourceModule("""
            #define MAXWOLVES """ + str(self.wolves) + """

            __device__ float xorshift32(unsigned int seed) {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                return (seed & 0x7FFFFFFF) / float(0x7FFFFFFF); // Normalizar a rango [0, 1]
            }
            __global__ void update(float *positions,
                                   float *wolves_positions,
                                   float a, int weights_number,
                                   float lower_bound, 
                                   float upper_bound,  
                                   unsigned int seed) {
                                   
                int thread = blockIdx.x * blockDim.x + threadIdx.x;
                
                if (thread < weights_number) {
                               
                    float A[MAXWOLVES];
                    float C[MAXWOLVES];
                    float next_position[MAXWOLVES];
                    float solution = 0.0;
                    unsigned int thread_seed = seed + thread;
                               
                    for (int i = 0; i < MAXWOLVES; i++){

                        float r1 = xorshift32(thread_seed + i);
                        float r2 = xorshift32(thread_seed + i + MAXWOLVES);
                               
                        A[i] = 2 * a * r1 - a;
                        C[i] = 2 * a * r2 - a;
                    }   
                               
                    for (int i = 0; i < MAXWOLVES; i++){
                               
                        float current_wolf_position = wolves_positions[thread + (weights_number * i)];
                        float prey_position = positions[thread];

                        next_position[i] = fabs(C[i] * current_wolf_position - prey_position);
                        float X = current_wolf_position - A[i] * next_position[i];
                        solution += X;
                    }
                               
                    solution /= MAXWOLVES;
                               
                    positions[thread] = solution;
                               
                    if(positions[thread] < lower_bound){

                        positions[thread] = lower_bound;            
                    }
                    else if(positions[thread] > upper_bound){
                               
                        positions[thread] = upper_bound;
                    }               
                }
            }
            """)

            # Inicializar y ejecutar el kernel
            update_positions = mod.get_function("update")
            block = 1024
            grid = (weights + block - 1) // block
            seed = self.get_seed()

            update_positions(positions_distance,
                            positions_distance_wolves,
                            np.float32(a),
                            np.int32(weights),
                            np.float32(self.lower_bound), 
                            np.float32(self.upper_bound),
                            seed, block=(block, 1, 1),
                            grid=(grid, 1))

            # Recuperamos los datos desde la GPU
            cuda.memcpy_dtoh(positions, positions_distance)
            self.round_positions[i] = positions

            final_time = time.time()
            print(f"Explotación {i + 1} / {self.agents} tiempo de ejecución: {final_time - inicial_time} segundos")

    def GWO_feature_explotation(self, epoch):

            for i in range(self.agents):

                inicial_time = time.time()
                weights = len(self.round_positions[0])

                a = 2 - epoch * (2 / self.epochs)

                positions = np.array(self.round_positions[i], dtype=np.float32)
                loss = np.array(self.loss, dtype=np.float32)
                round_positions = np.array(self.round_positions[i], dtype=np.float32)
                posicion = np.array(self.wolves_positions, dtype=np.float32)

                total_loss = sum(self.loss)

                for i in range(self.wolves):
                    loss[i] = self.loss[i] / total_loss
                
                # Alojar en  GPU

                positions_distance = cuda.mem_alloc(positions.nbytes)
                cuda.memcpy_htod(positions_distance, positions)

                normalized_loss = cuda.mem_alloc(self.loss.nbytes)
                cuda.memcpy_htod(normalized_loss, loss)

                positions_distance = cuda.mem_alloc(round_positions.nbytes)
                cuda.memcpy_htod(positions_distance, round_positions)

                positions_distance_wolves= cuda.mem_alloc(posicion.nbytes)
                cuda.memcpy_htod(positions_distance_wolves, posicion)

                mod = SourceModule("""
                #define MAXWOLVES """ + str(self.wolves) + """

                __device__ float xorshift32(unsigned int seed) {
                    seed ^= seed << 13;
                    seed ^= seed >> 17;
                    seed ^= seed << 5;
                    return (seed & 0x7FFFFFFF) / float(0x7FFFFFFF); // Normalizar a rango [0, 1]
                }
                __global__ void update(float *positions, 
                                       float *loss, 
                                       float *round_positions,
                                       float *wolves_positions, 
                                       float a,
                                        int weights_number, 
                                       float lower_bound, 
                                       float upper_bound, 
                                       unsigned int seed,
                                       unsigned int feature_probability,
                                       unsigned int signed_feature) {

                    int thread = blockIdx.x * blockDim.x + threadIdx.x;
                    
                    if (thread < weights_number) {
                                
                        float A[MAXWOLVES];
                        float C[MAXWOLVES];
                        float next_position[MAXWOLVES];
                        float solution = 0.0;
                        unsigned int thread_seed = seed + thread;    
                        
                        for (int i = 0; i < MAXWOLVES; i++){

                            unsigned int seed1 = thread_seed ^ (i * 2654435761U) ^ (thread_seed >> 13);
                            unsigned int seed2 = thread_seed ^ ((i + MAXWOLVES) * 2246822519U) ^ (thread_seed << 7);
                        
                            float r1 = xorshift32(seed1);
                            float r2 = xorshift32(seed2);
                                
                            A[i] = 2 * a * r1 - a;
                            C[i] = 2 * a * r2 - a;
                        }   
                                
                        for (int i = 0; i < MAXWOLVES; i++){
                                
                            float current_wolf_position = wolves_positions[thread + (weights_number * i)];
                            float prey_position = positions[thread];

                            next_position[i] = fabs(C[i] * current_wolf_position - prey_position);
                            float X = current_wolf_position - A[i] * next_position[i];
                            X *= loss[i];
                            solution += X;
                        }

                        solution /= MAXWOLVES;

                        float entropy_selection = xorshift32(feature_probability) * ((fabs(lower_bound) + fabs(upper_bound)) / 2);
                        float signed_entropy = xorshift32(signed_feature);

                        if(signed_entropy < 0.5){

                            entropy_selection *= -1;
                        }
                                
                        solution += entropy_selection;
                        positions[thread] = 1 / (1 + exp(-solution));
                        round_positions[thread] = positions[thread];
        
                        if(positions[thread] < lower_bound){

                            round_positions[thread] = lower_bound;            
                        }
                        else if(positions[thread] > upper_bound){
                                    
                            round_positions[thread] = upper_bound;
                        }    

                        if (positions[thread] < 0.5) {
                            positions[thread] = 0;
                        } 
                        else {
                            positions[thread] = 1;
                        }                     
                    }
                }
                """)

                # Inicializar y ejecutar el kernel
                update_positions = mod.get_function("update")
                block = 1024
                grid = (weights + block - 1) // block

                seed = self.get_seed()
                feature_probability = self.get_seed()
                signed_feature = self.get_seed()

                update_positions(positions_distance,
                                normalized_loss,
                                positions_distance,
                                positions_distance_wolves,
                                np.float32(a), 
                                np.int32(weights), 
                                np.float32(self.lower_bound),
                                np.float32(self.upper_bound),
                                seed, 
                                feature_probability,
                                signed_feature,
                                block=(block, 1, 1),
                                grid=(grid, 1))

                # Recuperamos los datos desde la GPU

                cuda.memcpy_dtoh(positions, positions_distance)
                cuda.memcpy_dtoh(round_positions, positions_distance)

                self.round_positions[i] = positions
                self.positions[i] = round_positions

                number_features = 0

                for feature in self.round_positions[i]:
                    if(feature == 1):
                        number_features += 1

                self.number_features[i] = number_features

                final_time = time.time()
                print(f"Explotación {i + 1} / {self.agents} tiempo de ejecución: {final_time - inicial_time} segundos")

    def get_seed(self):

        operative_sistem_entropy = int.from_bytes(os.urandom(4), byteorder='big')
        
        current_time = time.time_ns()
        hash_time = int(hashlib.sha256(str(current_time).encode()).hexdigest(), 16)
        
        random_number = np.random.randint(0, 2**32, dtype=np.uint32)
        
        semilla = (operative_sistem_entropy ^ hash_time ^ random_number) % (2**32)
        return np.uint32(semilla)
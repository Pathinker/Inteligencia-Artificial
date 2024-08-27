class Layer:

    def __init__(self):

        #Todas las capas estan compuestas por datos de entrada que reciben de capas previas y el output que emanan a la siguiente capa.

        self.input = None
        self.output = None

        def foward(self, input): 
            
            # Forward Propagation, actualiza el valor de las capas subsecuentes acorde el resultado obtenido

            pass

        def backward(self, outputGradient, learningRate): 
            
            # Backward Propagration es empleado para actualzar los parametros al ejecutarse el proceso de aprendizaje.
            # Sera usado el método de gradiente descente que toma en consideración el error del sistema respecto al output.
            # Gradiente Descendente: Busca Minimos, Gradiente Ascendente: Busca Máximos.
            # Él método de Gradiente busca llegar a derivadas 0, solamente cambia el signo conforme
            # learningRate: Es susceptible de incorporar un algoritmo optimizador.

            pass
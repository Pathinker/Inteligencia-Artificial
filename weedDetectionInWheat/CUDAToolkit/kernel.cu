
#include <curand_kernel.h>

extern "C" __global__ void actualizar(float *posiciones, float *posicionAlfa, float *posicionBeta, float *posicionDelta, float a, int num_pesos, int num_agentes, unsigned long long seed) {
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

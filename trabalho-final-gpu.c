%%writefile trabalho_final.cu

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand.h>
#include <curand_kernel.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <fstream>
#include <iostream>

using std::chrono::duration;
using std::cout;
using std::endl;
using myClock = std::chrono::high_resolution_clock;

#define N 2048
#define TAM_BLOCO 32
#define PI 3.14159265

//para checar erros chamadas Cuda
#define CUDA_SAFE_CALL(call) { \
   cudaError_t err = call;     \
   if(err != cudaSuccess) {    \
      fprintf(stderr,"Erro no arquivo '%s', linha %i: %s.\n",__FILE__, __LINE__,cudaGetErrorString(err)); \
      exit(EXIT_FAILURE); } } 

__global__ void calculate(unsigned long seed, curandState *state, double* results, 
    double initial_velocity, double timestep, double gravity) {

    int id = threadIdx.x;
    results[id] = -1;

    curand_init(seed, id, 0, &state[id]);

    double phi = (curand_normal(&state[id]) * 5 + 10) * (PI / 180.0);
    double theta = (curand_normal(&state[id]) * 3 + 15) * (PI / 180.0);

    double X[6];
    double X_prime[6];
    double X_1_E[6];
    double X_1[6];

    double init_vx, init_vy, init_vz;
    init_vx = initial_velocity * cos(theta) * sin(phi);
    init_vy = initial_velocity * cos(theta) * cos(phi);
    init_vz = initial_velocity * sin(theta);
    
    double init_dvx, init_dvy, init_dvz;
    init_dvx = 0;
    init_dvy = 0;
    init_dvz = -gravity;

    X[0] = 0;
    X[1] = 0;
    X[2] = 0;
    X[3] = init_vx;
    X[4] = init_vy;
    X[5] = init_vz;

    X_prime[0] = init_vx;
    X_prime[1] = init_vy;
    X_prime[2] = init_vz;
    X_prime[3] = init_dvx;
    X_prime[4] = init_dvy;
    X_prime[5] = init_dvz;

    for (int i = 0; i < 50; i++) {
        //Calculando passos
        X_1_E[0] = X[0] + (timestep * X_prime[0]);
        X_1_E[1] = X[1] + (timestep * X_prime[1]);
        X_1_E[2] = X[2] + (timestep * X_prime[2]);
        X_1_E[3] = X[3] + (timestep * X_prime[3]);
        X_1_E[4] = X[4] + (timestep * X_prime[4]);
        X_1_E[5] = X[5] + (timestep * X_prime[5]);

        X_1_E[0] = X_1_E[3];
        X_1_E[1] = X_1_E[4];
        X_1_E[2] = X_1_E[5];
        X_1_E[3] = 0;
        X_1_E[4] = 0;
        X_1_E[5] = -gravity;

        X_1[0] = X[0] + (timestep * ((X_prime[0] + X_1_E[0]) / 2.0));
        X_1[1] = X[1] + (timestep * ((X_prime[1] + X_1_E[1]) / 2.0));
        X_1[2] = X[2] + (timestep * ((X_prime[2] + X_1_E[2]) / 2.0));
        X_1[3] = X[3] + (timestep * ((X_prime[3] + X_1_E[3]) / 2.0));
        X_1[4] = X[4] + (timestep * ((X_prime[4] + X_1_E[4]) / 2.0));
        X_1[5] = X[5] + (timestep * ((X_prime[5] + X_1_E[5]) / 2.0));

        if (X_1[1] > 20.0) {
            double proportion = ((20.0 - X[1]) / (X_1[1] - X[1]));

            X[0] = X[0] + (X_1[0] - X[0]) * proportion;
            X[1] = X[1] + (X_1[1] - X[1]) * proportion;
            X[2] = X[2] + (X_1[2] - X[2]) * proportion;
            X[3] = X[3] + (X_1[3] - X[3]) * proportion;
            X[4] = X[4] + (X_1[4] - X[4]) * proportion;
            X[5] = X[5] + (X_1[5] - X[5]) * proportion;

            double stoptime = ((i-1) * timestep + (proportion) * timestep);
            results[id] = stoptime;

            break;
        }

        //Deixando pronto para próxima iteração
        X[0] = X_1[0];
        X[1] = X_1[1];
        X[2] = X_1[2];
        X[3] = X_1[3];
        X[4] = X_1[4];
        X[5] = X_1[5];

        X_prime[0] = X[3];
        X_prime[1] = X[4];
        X_prime[2] = X[5];
        X_prime[3] = init_dvx;
        X_prime[4] = init_dvy;
        X_prime[5] = init_dvz;
    }
}

//funcao principal
int main(int argc, char** argv) {

    int n_blocos = 1;
    int n_threads = 8;

    double* h_results;
    double* d_results;

    h_results = (double*) malloc(sizeof(double) * n_threads);

    //Initialization
    double initial_velocity = 25.0;
    double timestep = 0.1;
    double gravity = 9.81;
    // double object_mass = 0.425;
    // double object_radius = 0.111;
    // double air_density = 1.2;
    // double drag_coeff_a = 0.214;
    // double drag_coeff_b = 0.283;
    // double drag_coeff_vc = 7.5;
    // double drag_coeff_vs = 0.707;
    // double object_angular_velocity = 6.28;
    // double proportionality_constant = 1;

    int n_bytes = sizeof(double) * n_threads;
    CUDA_SAFE_CALL(cudaMalloc((void**) &d_results, n_bytes));

    curandState *devStates;
    CUDA_SAFE_CALL(cudaMalloc((void**) &devStates, n_blocos * n_threads * sizeof(curandState)));

    calculate <<<n_blocos, n_threads>>>(time(NULL), devStates, d_results, initial_velocity, timestep, gravity);
    CUDA_SAFE_CALL(cudaGetLastError());

    CUDA_SAFE_CALL(cudaMemcpy(h_results, d_results, n_bytes, cudaMemcpyDeviceToHost)) 

    for (int i = 0; i < n_threads; i++) {
        printf("RESULT[%d]: %lf\n", i, h_results[i]);
    }

	return 0;
}
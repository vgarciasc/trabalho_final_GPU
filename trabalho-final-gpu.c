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

__device__ void differentiate_X(double* X, double* X_prime, double drag_coeff, double drag_coeff_a, double drag_coeff_b, 
    double drag_coeff_vc, double drag_coeff_vs, double magnus_coeff, double gravity) {

    double vx = X[3];
    double vy = X[4];
    double vz = X[5];
    double velocity = sqrt(vx * vx + vy * vy + vz * vz);
    double sqrt_vx_vy = sqrt(vx * vx + vy * vy);
    double drag_coeff_v = drag_coeff_a + (drag_coeff_b / (1 + exp(((velocity - drag_coeff_vc) / drag_coeff_vs))));

    X_prime[0] = X[3];
    X_prime[1] = X[4];
    X_prime[2] = X[5];
    X_prime[3] = - (drag_coeff) * ((drag_coeff_v * velocity * vx) + ((magnus_coeff * velocity * vy) / (sqrt_vx_vy)));
    X_prime[4] = - (drag_coeff) * ((drag_coeff_v * velocity * vy) - ((magnus_coeff * velocity * vx) / (sqrt_vx_vy)));
    X_prime[5] = - (drag_coeff) * (drag_coeff_v * velocity * vz) - gravity;
}

__global__ void calculate(unsigned long seed, curandState *state, double* results, int result_size,
    double velocity, double timestep, double gravity, double drag_coeff,
    double drag_coeff_a, double drag_coeff_b, double drag_coeff_vc, double drag_coeff_vs,
    double object_radius, double object_angular_velocity, double proportionality_constant) {

    int id = threadIdx.x;
    curand_init(seed, id, 0, &state[id]);

    //Usar a mesma Seed pros dois pode corromper nossas amostras?
    double phi = (curand_normal(&state[id]) * 5 + 10) * (PI / 180.0);
    double theta = (curand_normal(&state[id]) * 3 + 15) * (PI / 180.0);

    double X[6];
    double X_prime[6];
    double X_1_E[6];
    double X_1[6];

    double vx, vy, vz;
    vx = velocity * cos(theta) * sin(phi);
    vy = velocity * cos(theta) * cos(phi);
    vz = velocity * sin(theta);
    double sqrt_vx_vy = sqrt(vx * vx + vy * vy);

    //X = (x, y, z, vx, vy, vz)
    X[0] = 0;
    X[1] = 0;
    X[2] = 0;
    X[3] = vx;
    X[4] = vy;
    X[5] = vz;

    //drag_coeff_v = Cr(v) = a + b/(1 + exp((v - vc) / vs))
    double drag_coeff_v = drag_coeff_a + (drag_coeff_b / (1 + exp(((velocity - drag_coeff_vc) / drag_coeff_vs))));
    //magnus_coeff = Cm * r * w
    double magnus_coeff = proportionality_constant * object_angular_velocity * object_radius;

    //X' = (vx, vy, vz, dvx, dvy, dvz)
    differentiate_X(X, X_prime, drag_coeff, drag_coeff_a, drag_coeff_b, drag_coeff_vc, drag_coeff_vs, magnus_coeff, gravity);

    for (int i = 0; i < 500; i++) {
        // if (id == 0) {
        //     printf(" time: %lf,\n", i * timestep);
        //     printf(" x*: %lf,\n", X[0]);
        //     printf(" y*: %lf,\n", X[1]);
        //     printf(" z*: %lf,\n", X[2]);
        //     printf(" vx*: %lf,\n", X[3]);
        //     printf(" vy*: %lf,\n", X[4]);
        //     printf(" vz*: %lf,\n\n\n", X[5]);
        // }

        //=== Calculando passos

        //Xe(n+1) = X(n) + Δt*X'(n)
        X_1_E[0] = X[0] + (timestep * X_prime[0]);
        X_1_E[1] = X[1] + (timestep * X_prime[1]);
        X_1_E[2] = X[2] + (timestep * X_prime[2]);
        X_1_E[3] = X[3] + (timestep * X_prime[3]);
        X_1_E[4] = X[4] + (timestep * X_prime[4]);
        X_1_E[5] = X[5] + (timestep * X_prime[5]);

        //X'e(n+1) armazenado em Xe(n+1)
        differentiate_X(X_1_E, X_1_E, drag_coeff, drag_coeff_a, drag_coeff_b, drag_coeff_vc, drag_coeff_vs, magnus_coeff, gravity);

        //X(n+1) = X + (Δt/2)*(X'(n) + X'e(n+1))
        X_1[0] = X[0] + (timestep * ((X_prime[0] + X_1_E[0]) / 2.0));
        X_1[1] = X[1] + (timestep * ((X_prime[1] + X_1_E[1]) / 2.0));
        X_1[2] = X[2] + (timestep * ((X_prime[2] + X_1_E[2]) / 2.0));
        X_1[3] = X[3] + (timestep * ((X_prime[3] + X_1_E[3]) / 2.0));
        X_1[4] = X[4] + (timestep * ((X_prime[4] + X_1_E[4]) / 2.0));
        X_1[5] = X[5] + (timestep * ((X_prime[5] + X_1_E[5]) / 2.0));

        if (X_1[1] > 20.0) {
            double proportion = ((20.0 - X[1]) / (X_1[1] - X[1]));

            results[id * result_size + 0] = X[0] + (X_1[0] - X[0]) * proportion;
            results[id * result_size + 1] = X[1] + (X_1[1] - X[1]) * proportion;
            results[id * result_size + 2] = X[2] + (X_1[2] - X[2]) * proportion;
            results[id * result_size + 3] = X[3] + (X_1[3] - X[3]) * proportion;
            results[id * result_size + 4] = X[4] + (X_1[4] - X[4]) * proportion;
            results[id * result_size + 5] = X[5] + (X_1[5] - X[5]) * proportion;

            double stoptime = ((i-1) * timestep + (proportion) * timestep);
            results[id * result_size + 6] = stoptime;

            //talvez ela devesse continuar calculando, pra ficar junto das irmãs?
            break;
        }

        //=== Deixando pronto para próxima iteração

        //X(n) <- X(n+1)
        X[0] = X_1[0];
        X[1] = X_1[1];
        X[2] = X_1[2];
        X[3] = X_1[3];
        X[4] = X_1[4];
        X[5] = X_1[5];
        
        differentiate_X(X, X_prime, drag_coeff, drag_coeff_a, drag_coeff_b, drag_coeff_vc, drag_coeff_vs, magnus_coeff, gravity);
    }
}

int main(int argc, char** argv) {

    int n_blocos = 1;
    int n_threads = 8;
    int result_size = 7;
    int n_bytes = sizeof(double) * n_threads * result_size;

    double* h_results;
    double* d_results;

    h_results = (double*) malloc(n_bytes);

    //Initialization
    double initial_velocity = 25.0;
    double timestep = 0.1;
    double gravity = 9.81;

    double drag_coeff;
    double object_mass = 0.425;
    double object_radius = 0.111;
    double air_density = 1.2;

    drag_coeff = (air_density * (PI * (object_radius * object_radius))) / (2.0 * object_mass); 
    
    double drag_coeff_a = 0.214;
    double drag_coeff_b = 0.283;
    double drag_coeff_vc = 7.5;
    double drag_coeff_vs = 0.707;

    double object_angular_velocity = 6.28;
    double proportionality_constant = 1;

    CUDA_SAFE_CALL(cudaMalloc((void**) &d_results, n_bytes));

    curandState *devStates;
    CUDA_SAFE_CALL(cudaMalloc((void**) &devStates, n_blocos * n_threads * sizeof(curandState)));

    calculate<<<n_blocos, n_threads>>>(time(NULL), devStates, d_results, result_size, initial_velocity, 
        timestep, gravity, drag_coeff, drag_coeff_a, drag_coeff_b, drag_coeff_vc, drag_coeff_vs,
        object_radius, object_angular_velocity, proportionality_constant);
    CUDA_SAFE_CALL(cudaGetLastError());

    CUDA_SAFE_CALL(cudaMemcpy(h_results, d_results, n_bytes, cudaMemcpyDeviceToHost));

    double t_acc = 0.0;
    double x_acc = 0.0;
    double y_acc = 0.0;
    double z_acc = 0.0;

    for (int i = 0; i < n_threads; i++) {
        t_acc += h_results[i * result_size + 6];
        x_acc += h_results[i * result_size + 0];
        y_acc += h_results[i * result_size + 1];
        z_acc += h_results[i * result_size + 2];
    }

    printf("RESULTS: \n");
    printf(" E[tf] = %lf\n", (t_acc / n_threads));
    printf(" E[x*] = %lf\n", (x_acc / n_threads));
    printf(" E[y*] = %lf\n", (y_acc / n_threads));
    printf(" E[z*] = %lf\n", (z_acc / n_threads));

	return 0;
}
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
#include <sys/time.h>

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

//argumento deve ser double
#define GET_TIME(now) { \
   struct timespec time; \
   clock_gettime(CLOCK_MONOTONIC_RAW, &time); \
   now = time.tv_sec + time.tv_nsec/1000000000.0; \
}

__device__ void differentiate_X(double* X_src, double* X_dst, double drag_coeff, double drag_coeff_a, double drag_coeff_b, 
    double drag_coeff_vc, double drag_coeff_vs, double magnus_coeff, double gravity) {

    double vx = X_src[3];
    double vy = X_src[4];
    double vz = X_src[5];
    double velocity = sqrt(vx * vx + vy * vy + vz * vz);
    double sqrt_vx_vy = sqrt(vx * vx + vy * vy);
    double drag_coeff_v = drag_coeff_a + (drag_coeff_b / (1 + exp(((velocity - drag_coeff_vc) / drag_coeff_vs))));

    X_dst[0] = vx;
    X_dst[1] = vy;
    X_dst[2] = vz;
    X_dst[3] = - (drag_coeff) * ((drag_coeff_v * velocity * vx) + ((magnus_coeff * velocity * vy) / (sqrt_vx_vy)));
    X_dst[4] = - (drag_coeff) * ((drag_coeff_v * velocity * vy) - ((magnus_coeff * velocity * vx) / (sqrt_vx_vy)));
    X_dst[5] = - (drag_coeff) * (drag_coeff_v * velocity * vz) - gravity;
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

    //X = (0, 0, 0, init_vx, init_vy, init_vz)
    X[0] = 0;
    X[1] = 0;
    X[2] = 0;
    X[3] = velocity * cos(theta) * sin(phi);
    X[4] = velocity * cos(theta) * cos(phi);
    X[5] = velocity * sin(theta);

    //drag_coeff_v = Cr(v) = a + b/(1 + exp((v - vc) / vs))
    double drag_coeff_v = drag_coeff_a + (drag_coeff_b / (1 + exp(((velocity - drag_coeff_vc) / drag_coeff_vs))));
    //magnus_coeff = Cm * r * w
    double magnus_coeff = proportionality_constant * object_angular_velocity * object_radius;

    for (int i = 0; i < 500; i++) {
        //X' = (vx, vy, vz, dvx, dvy, dvz)
        differentiate_X(X, X_prime, drag_coeff, drag_coeff_a, drag_coeff_b, drag_coeff_vc, drag_coeff_vs, magnus_coeff, gravity);

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
        X[0] += (timestep * ((X_prime[0] + X_1_E[0]) / 2.0));
        X[1] += (timestep * ((X_prime[1] + X_1_E[1]) / 2.0));
        X[2] += (timestep * ((X_prime[2] + X_1_E[2]) / 2.0));
        X[3] += (timestep * ((X_prime[3] + X_1_E[3]) / 2.0));
        X[4] += (timestep * ((X_prime[4] + X_1_E[4]) / 2.0));
        X[5] += (timestep * ((X_prime[5] + X_1_E[5]) / 2.0));

        if (X[1] > 20.0) {
            //=== Interpolando resultados
            double end_value = X[1];
            double previous_value = X[1] - (timestep * ((X_prime[1] + X_1_E[1]) / 2.0));
            double proportion = 1 - (end_value - 20) / (end_value - previous_value);

            results[id * result_size + 0] = X[0] + ((timestep * ((X_prime[0] + X_1_E[0]) / 2.0))) * (proportion - 1);
            results[id * result_size + 1] = X[1] + ((timestep * ((X_prime[1] + X_1_E[1]) / 2.0))) * (proportion - 1);
            results[id * result_size + 2] = X[2] + ((timestep * ((X_prime[2] + X_1_E[2]) / 2.0))) * (proportion - 1);
            results[id * result_size + 3] = X[3] + ((timestep * ((X_prime[3] + X_1_E[3]) / 2.0))) * (proportion - 1);
            results[id * result_size + 4] = X[4] + ((timestep * ((X_prime[4] + X_1_E[4]) / 2.0))) * (proportion - 1);
            results[id * result_size + 5] = X[5] + ((timestep * ((X_prime[5] + X_1_E[5]) / 2.0))) * (proportion - 1);

            double stoptime = ((i-1) * timestep + (proportion) * timestep);
            results[id * result_size + 6] = stoptime;

            //talvez ela devesse continuar calculando, pra ficar junto das irmãs?
            break;
        }
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

    double inicio, fim;
    double tempo_media;
    cudaEvent_t start, stop;

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

    CUDA_SAFE_CALL(cudaEventCreate(&start));
    CUDA_SAFE_CALL(cudaEventCreate(&stop));
    CUDA_SAFE_CALL(cudaEventRecord(start));
    calculate<<<n_blocos, n_threads>>>(time(NULL), devStates, d_results, result_size, initial_velocity, 
        timestep, gravity, drag_coeff, drag_coeff_a, drag_coeff_b, drag_coeff_vc, drag_coeff_vs,
        object_radius, object_angular_velocity, proportionality_constant);
    CUDA_SAFE_CALL(cudaGetLastError());
    CUDA_SAFE_CALL(cudaEventRecord(stop));
    CUDA_SAFE_CALL(cudaEventSynchronize(stop));
    float delta_eventos = 0;
    CUDA_SAFE_CALL(cudaEventElapsedTime(&delta_eventos, start, stop));

    CUDA_SAFE_CALL(cudaMemcpy(h_results, d_results, n_bytes, cudaMemcpyDeviceToHost));

    double t_acc = 0.0;
    double x_acc = 0.0;
    double y_acc = 0.0;
    double z_acc = 0.0;
    int goals = 0;

    GET_TIME(inicio);  
    for (int i = 0; i < n_threads; i++) {
        t_acc += h_results[i * result_size + 6];
        x_acc += h_results[i * result_size + 0];
        y_acc += h_results[i * result_size + 1];
        z_acc += h_results[i * result_size + 2];

        if (h_results[i * result_size + 2] > 0.0   && h_results[i * result_size + 2] < 2.44 &&
            h_results[i * result_size + 2] > -3.66 && h_results[i * result_size + 2] < 3.66) {
            goals++;
        }
    }
    GET_TIME(fim);  
    tempo_media = fim - inicio;

    printf("RESULTS: \n");
    printf("  E[tf] = %lf\n", (t_acc / n_threads));
    printf("  E[x*] = %lf\n", (x_acc / n_threads));
    printf("  E[y*] = %lf\n", (y_acc / n_threads));
    printf("  E[z*] = %lf\n", (z_acc / n_threads));
    printf("  goal rate = %lf%%\n", ((double) goals / (n_threads) * 100.0));

    printf("DURATION: \n");
    printf("  Kernel duration = %f seg \n", delta_eventos / 1000);
    printf("  Mean calc duration = %f seg \n", tempo_media);

	return 0;
}
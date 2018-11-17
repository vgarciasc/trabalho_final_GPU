%%writefile trabalho_final.cu

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

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

double to_radians(double degrees) {
    return degrees * (PI / 180.0);
}

void print_X(double* X) {
    printf("X: [\n x: %lf,\n y: %lf,\n z: %lf,\n vx: %lf,\n vy: %lf,\n vz: %lf\n]\n", X[0], X[1], X[2], X[3], X[4], X[5]);
}

//funcao principal
int main(int argc, char** argv) {

    //Initialization
    double initial_velocity;
    double theta;
    double phi;
    double timestep = 0.1;
    double gravity = 9.81;
    
    sscanf(argv[1], "%lf", &initial_velocity);
    sscanf(argv[2], "%lf", &theta);
    sscanf(argv[3], "%lf", &phi);

    theta = to_radians(theta);
    phi = to_radians(phi);
    
    double* X;
    X = (double*) malloc(sizeof(double) * 6);
    
    X[0] = 0;
    X[1] = 0;
    X[2] = 0;
    X[3] = initial_velocity * cos(theta) * sin(phi);
    X[4] = initial_velocity * cos(theta) * cos(phi);
    X[5] = initial_velocity * sin(theta);

    double* X_prime;
    X_prime = (double*) malloc(sizeof(double) * 6);
    
    X_prime[0] = X[3];
    X_prime[1] = X[4];
    X_prime[2] = X[5];
    X_prime[3] = 0;
    X_prime[4] = 0;
    X_prime[5] = -gravity;

    double* X_1_E;
    X_1_E = (double*) malloc(sizeof(double) * 6);

    double* X_1;
    X_1 = (double*) malloc(sizeof(double) * 6);

    printf("TIME START:\n");
    print_X(X);

    for (int i = 0; i < 15; i++) {
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
            printf("==== PASSED GOAL! ====\n");
            printf("> TIME %lf:\n", ((i-1) * timestep));
            print_X(X);
            printf("> TIME %lf:\n", (i * timestep));
            print_X(X_1);

            printf("> INTERPOLATING...\n");
            double proportion = ((20.0 - X[1]) / (X_1[1] - X[1]));
            printf("> PROPORTION: %lf\n", proportion);

            X[0] = X[0] + (X_1[0] - X[0]) * proportion;
            X[1] = X[1] + (X_1[1] - X[1]) * proportion;
            X[2] = X[2] + (X_1[2] - X[2]) * proportion;
            X[3] = X[3] + (X_1[3] - X[3]) * proportion;
            X[4] = X[4] + (X_1[4] - X[4]) * proportion;
            X[5] = X[5] + (X_1[5] - X[5]) * proportion;

            printf("> TIME %lf:\n", ((i-1) * timestep + (proportion) * timestep));
            print_X(X);

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
        X_prime[3] = 0;
        X_prime[4] = 0;
        X_prime[5] = -gravity;

        // printf("TIME %lf:\n", (i * timestep));
        // print_X(X);
    }

	return 0;
}
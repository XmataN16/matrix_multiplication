#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void init_matrix(int N, int M, int K, double* A, double* B, double* C);
void print_matrix(int N, int M, double* matrix);
void blas_dgemm(int N, int M, int K, double* A, double* B, double* C);

int main(void)
{
    /* initialization of matrix sizes */
    const int N = 1000;
    const int M = 1000;
    const int K = 1000;

    /* allocation of memory for arrays A, B and result matrix C */
    double* A = (double*)malloc(N * M * sizeof(double));
    double* B = (double*)malloc(M * K * sizeof(double));
    double* C = (double*)malloc(N * K * sizeof(double));

    if (!A || !B || !C) 
    {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    init_matrix(N, M, K, A, B, C);

    double start_time = omp_get_wtime();
    blas_dgemm(N, M, K, A, B, C);
    double end_time = omp_get_wtime();

    //print_matrix(N, M, C);

    printf("Execution time: %f seconds\n", end_time - start_time);

    /* free allocated memory */
    free(A);
    free(B);
    free(C);

    return 0;
}

void blas_dgemm(int N, int M, int K, double* A, double* B, double* C)
{
    int i, j, k;
    for (i = 0; i < N; i++) 
    {
        double* c = C + i * M;
        for (k = 0; k < K; k++) 
        {
            const double* b = B + k * M;
            double a = A[i * K + k];
            for (j = 0; j < M; j++) 
            {
                c[j] += a * b[j];
            }
        }
    }
}

/* Function for displaying the matrix to the console */
void print_matrix(int N, int M, double* matrix)
{
    int i, j;
    for (i = 0; i < N; i++) 
    {
        for (j = 0; j < M; j++) 
        {
            printf("%f\t", matrix[i * N + j]);
        }
        printf("\n");
    }
}

/* The initialization function of matrices A, B and C */
void init_matrix(int N, int M, int K, double* A, double* B, double* C)
{
    int i, j;

    /* Initialization of matrix A (N x M) */
    for (i = 0; i < N; i++) 
    {
        for (j = 0; j < M; j++) 
        {
            A[i * M + j] = 1.0;
        }
    }

    /* Initialization of matrix B (M x K) */
    for (i = 0; i < M; i++) 
    {
        for (j = 0; j < K; j++) 
        {
            B[i * K + j] = 1.0;
        }
    }

    /* Initialization of matrix C (N x K) */
    for (i = 0; i < N; i++) 
    {
        for (j = 0; j < K; j++) 
        {
            C[i * K + j] = 0.0;
        }
    }
}
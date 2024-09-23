#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <omp.h>
#include <immintrin.h>

void init_matrix(int N, int M, int K, double* A, double* B, double* C);
void print_matrix(int N, int M, double* matrix);
void blas_dgemm(int N, int M, int K, double* A, double* B, double* C);

int main(int argc, char** argv)
{
    /* Declaration variables of matrix sizes */
    int N, M, K;
    
    /* Default value threads count */
    int num_threads;

    /* Checking the number of command line arguments */
    if (argc == 3) 
    {
        /* Converting strings to an integer */
        num_threads = atoi(argv[1]);
        /* Initialization of matrix sizes */
        N = M = K = atoi(argv[2]); 
    }
    else if (argc == 2)
    {
        num_threads = atoi(argv[1]);
        N = M = K = 1000;
    }
    else if (argc == 1)
    {
        num_threads = omp_get_max_threads();
        N = M = K = 1000;
    }
    /* Set the number of threads for OpenMP */
    omp_set_num_threads(num_threads);

    /* Allocation of memory for arrays A, B and result matrix C */
    double* A = (double*)malloc(N * M * sizeof(double));
    double* B = (double*)malloc(M * K * sizeof(double));
    double* C = (double*)malloc(N * K * sizeof(double));

    if (!A || !B || !C) 
    {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    init_matrix(N, M, K, A, B, C);

    /* The start of the execution time */
    double start_time = omp_get_wtime();

    blas_dgemm(N, M, K, A, B, C);

    /* The end of the execution time */
    double end_time = omp_get_wtime();

    //print_matrix(N, M, C);

    printf("Execution time: %f seconds\n", end_time - start_time);

    /* Free allocated memory */
    free(A);
    free(B);
    free(C);

    return 0;
}

/* Own implementation of the matrix multiplication function */
void blas_dgemm(int N, int M, int K, double* A, double* B, double* C)
{
    int i, j, k;

    #pragma omp parallel for private(i,j,k) schedule(static)
    for (i = 0; i < N; i++)
    {
        double* c = C + i * M;
        for (k = 0; k < K; k++)
        {
            const double* b = B + k * M;
            double a = A[i * K + k];

            /* Loading a into a vector register */
            __m256d a_vec = _mm256_set1_pd(a);

            /* Vector cycle to speed up operations */ 
            for (j = 0; j <= M - 4; j += 4)
            {
                __m256d c_vec = _mm256_loadu_pd(&c[j]);
                __m256d b_vec = _mm256_loadu_pd(&b[j]);
                /* C += A * B */
                c_vec = _mm256_fmadd_pd(a_vec, b_vec, c_vec); 
                _mm256_storeu_pd(&c[j], c_vec);
            }

            /* The remaining elements if M is not divisible by 4 */ 
            for (; j < M; j++)
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
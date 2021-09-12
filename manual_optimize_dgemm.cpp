#include <cstdio>
#include <algorithm>
#include "halide_benchmark.h"
#include "halide_macros.h"

void naive_dgemm(const double *A, const double *B, double *C, const uint32_t M, const uint32_t N, const uint32_t K)
{
    for (uint32_t i = 0; i < M; i++)
    {
        for (uint32_t j = 0; j < N; j++)
        {
            double sum = 0.0;
            for (uint32_t k = 0; k < K; k++)
                sum += A[K * i + k] * B[N * k + j];
            C[N * i + j] = sum;
        }
    }
}

double *transpose(const double *B, const uint32_t K, const uint32_t N)
{
    double *Bb = (double *)malloc(K * N * sizeof(double));
    for (uint32_t n = 0; n < N; ++n)
    {
        for (uint32_t k = 0; k < K; ++k)
        {
            Bb[n * K + k] = B[k * N + n];
        }
    }
    return Bb;
}

void manual_dgemm(const double *A, const double *B, double *C, const uint32_t M, const uint32_t N, const uint32_t K)
{
    const uint32_t TILE_H = 32;
    const uint32_t TILE_W = 32;

    double *Bb = transpose(B, K, N);
    for (uint32_t i = 0; i < M; i++)
    {
        for (uint32_t j = 0; j < N; j++)
        {
            double sum = 0.0;
            for (uint32_t k = 0; k < K; k++)
                sum += A[K * i + k] * Bb[K * j + k];
            C[N * i + j] = sum;
        }
    }
    free(Bb);
}

int main()
{
    double *A, *B, *C, *CRef;
    int m, n, k, i, j;
    double alpha, beta;

    printf("\n This example computes real matrix C=alpha*A*B+beta*C using \n"
           " Intel(R) MKL function dgemm, where A, B, and  C are matrices and \n"
           " alpha and beta are double precision scalars\n\n");

    m = 2000, k = 200, n = 1000;
    printf(" Initializing data for matrix multiplication C=A*B for matrix \n"
           " A(%ix%i) and matrix B(%ix%i)\n\n",
           m, k, k, n);
    alpha = 1.0;
    beta = 0.0;

    printf(" Allocating memory for matrices aligned on 64-byte boundary for better \n"
           " performance \n\n");
    A = (double *)malloc(m * k * sizeof(double));
    B = (double *)malloc(k * n * sizeof(double));
    C = (double *)malloc(m * n * sizeof(double));
    CRef = (double *)malloc(m * n * sizeof(double));

    if (A == NULL || B == NULL || C == NULL || CRef == NULL)
    {
        printf("\n ERROR: Can't allocate memory for matrices. Aborting... \n\n");
        free(A);
        free(B);
        free(C);
        free(CRef);
        return 1;
    }

    printf(" Intializing matrix data \n\n");
    for (i = 0; i < (m * k); i++)
    {
        A[i] = (double)(i + 1);
    }

    for (i = 0; i < (k * n); i++)
    {
        B[i] = (double)(-i - 1);
    }

    for (i = 0; i < (m * n); i++)
    {
        C[i] = 0.0;
        CRef[i] = 0.0;
    }

    set_math_flags();
    naive_dgemm(A, B, CRef, m, n, k);

    printf(" Computing matrix product using Intel(R) MKL dgemm function via CBLAS interface \n\n");
    manual_dgemm(A, B, C, m, n, k);

    for (i = 0; i < (m * n); i++)
    {
        if (abs(C[i] - CRef[i]) > 1e-4)
        {
            printf("error at %d\n", i);
            return 1;
        }
    }

    manual_dgemm(A, B, C, m, n, k);
    printf("\n Computations completed.\n\n");
    double elapsed = 1e6 * Halide::Tools::benchmark([&]()
                                                    { manual_dgemm(A, B, C, m, n, k); });
    printf("time(us): %f, gflops: %f\n", elapsed, m * n * k * 2 * 1e-3 / elapsed);

    printf(" Top left corner of matrix A: \n");
    for (i = 0; i < std::min(m, 6); i++)
    {
        for (j = 0; j < std::min(k, 6); j++)
        {
            printf("%12.0f", A[j + i * k]);
        }
        printf("\n");
    }

    printf("\n Top left corner of matrix B: \n");
    for (i = 0; i < std::min(k, 6); i++)
    {
        for (j = 0; j < std::min(n, 6); j++)
        {
            printf("%12.0f", B[j + i * n]);
        }
        printf("\n");
    }

    printf("\n Top left corner of matrix C: \n");
    for (i = 0; i < std::min(m, 6); i++)
    {
        for (j = 0; j < std::min(n, 6); j++)
        {
            printf("%12.5G", C[j + i * n]);
        }
        printf("\n");
    }

    printf("\n Deallocating memory \n\n");
    free(A);
    free(B);
    free(C);

    printf(" Example completed. \n\n");
    return 0;
}
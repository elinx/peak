#include <algorithm>
#include <cstdio>
#include <cstring>

#include "halide_benchmark.h"
#include "halide_macros.h"

void naive_dgemm(const double *A, const double *B, double *C, const uint32_t M, const uint32_t N,
                 const uint32_t K) {
  for (uint32_t i = 0; i < M; i++) {
    for (uint32_t j = 0; j < N; j++) {
      double sum = 0.0;
      for (uint32_t k = 0; k < K; k++) sum += A[K * i + k] * B[N * k + j];
      C[N * i + j] = sum;
    }
  }
}

template <uint32_t k_inner_bound, uint32_t n_inner_bound, uint32_t n_inner_step>
static void PackB(double *Bp, const double *B, uint32_t K, uint32_t N, uint32_t k_outer,
                  uint32_t n_outer) {
  memset(Bp, 0, sizeof(double) * k_inner_bound * n_inner_bound);
  uint32_t k_start = k_outer;
  uint32_t k_end = std::min(K, k_start + k_inner_bound);
  uint32_t n_start = n_outer;
  uint32_t n_end = std::min(N, n_start + n_inner_bound);
  for (uint32_t k = k_start; k < k_end; ++k) {
    for (uint32_t n = n_start; n < n_end; ++n) {
      uint32_t r = (n - n_start) / n_inner_step;
      uint32_t rr = (n - n_start) % n_inner_step;
      uint32_t c = k - k_start;
      uint32_t i = r * n_inner_step * k_inner_bound + c * n_inner_step + rr;
      Bp[i] = B[k * N + n];
    }
  }
}

template <uint32_t m_inner_bound, uint32_t k_inner_bound, uint32_t m_inner_step>
static void PackA(double *Ap, const double *A, uint32_t M, uint32_t K, uint32_t m_outer,
                  uint32_t k_outer) {
  memset(Ap, 0, sizeof(double) * k_inner_bound * m_inner_bound);
  uint32_t m_start = m_outer;
  uint32_t m_end = std::min(M, m_start + m_inner_bound);
  uint32_t k_start = k_outer;
  uint32_t k_end = std::min(K, k_start + k_inner_bound);
  for (uint32_t k = k_start; k < k_end; ++k) {
    for (uint32_t m = m_start; m < m_end; ++m) {
      uint32_t r = (m - m_start) / m_inner_step;
      uint32_t rr = (m - m_start) % m_inner_step;
      uint32_t c = k - k_start;
      uint32_t i = r * m_inner_step * k_inner_bound + c * m_inner_step + rr;
      Ap[i] = A[m * K + k];
    }
  }
}

template <uint32_t m_inner_bound, uint32_t n_inner_bound>
static void WriteBackC(const double *Cc, double *C, uint32_t M, uint32_t N, uint32_t m_outer,
                       uint32_t n_outer) {
  for (uint32_t m = 0; m < m_inner_bound; ++m) {
    for (uint32_t n = 0; n < n_inner_bound; ++n) {
      uint32_t C_m = m_outer + m;
      uint32_t C_n = n_outer + n;
      if (C_m < M && C_n < N) {
        C[C_m * N + C_n] += Cc[m * n_inner_bound + n];
      }
    }
  }
}

template <uint32_t M, uint32_t N, uint32_t K, uint32_t CN>
static inline void micro_kernel(const double *A, const double *B, double *C) {
  for (uint32_t k = 0; k < K; ++k) {
    for (uint32_t m = 0; m < M; ++m) {
      for (uint32_t n = 0; n < N; ++n) {
        C[m * CN + n] += A[k * M + m] * B[k * N + n];
      }
    }
  }
}

void manual_dgemm(const double *A, const double *B, double *C, const uint32_t M, const uint32_t N,
                  const uint32_t K) {
  constexpr uint32_t TILE_H = 8;
  constexpr uint32_t TILE_W = 8;
  constexpr uint32_t TILE_K = 128;

  constexpr uint32_t m_outer_step = TILE_H * 8;
  constexpr uint32_t n_outer_step = TILE_W * 16;
  constexpr uint32_t k_outer_step = TILE_K;

  const uint32_t m_outer_bound = (M + m_outer_step - 1) / m_outer_step * m_outer_step;
  const uint32_t n_outer_bound = (N + n_outer_step - 1) / n_outer_step * n_outer_step;
  const uint32_t k_outer_bound = (K + n_outer_step - 1) / n_outer_step * n_outer_step;

  constexpr uint32_t m_inner_bound = m_outer_step;
  constexpr uint32_t n_inner_bound = n_outer_step;
  constexpr uint32_t k_inner_bound = k_outer_step;

  constexpr uint32_t m_inner_step = TILE_H;
  constexpr uint32_t n_inner_step = TILE_W;

  double *Bc = (double *)aligned_alloc(32, sizeof(double) * k_inner_bound * n_inner_bound);
  double *Ac = (double *)aligned_alloc(32, sizeof(double) * m_inner_bound * k_inner_bound);
  double *Cc = (double *)aligned_alloc(32, sizeof(double) * m_inner_bound * n_inner_bound);
  memset(C, 0, M * N * sizeof(double));

  for (uint32_t n_outer = 0; n_outer < n_outer_bound; n_outer += n_outer_step) {
    for (uint32_t k_outer = 0; k_outer < k_outer_bound; k_outer += k_outer_step) {
      PackB<k_inner_bound, n_inner_bound, n_inner_step>(Bc, B, K, N, k_outer, n_outer);
      for (uint32_t m_outer = 0; m_outer < m_outer_bound; m_outer += m_outer_step) {
        PackA<m_inner_bound, k_inner_bound, m_inner_step>(Ac, A, M, K, m_outer, k_outer);
        memset(Cc, 0, sizeof(double) * m_inner_bound * n_inner_bound);
        for (uint32_t n_inner = 0; n_inner < n_inner_bound; n_inner += n_inner_step) {
          for (uint32_t m_inner = 0; m_inner < m_inner_bound; m_inner += m_inner_step) {
            micro_kernel<m_inner_step, n_inner_step, k_inner_bound, n_inner_bound>(
                &Ac[m_inner * k_inner_bound], &Bc[n_inner * k_inner_bound],
                &Cc[m_inner * n_inner_bound + n_inner]);
          }
        }
        WriteBackC<m_inner_bound, n_inner_bound>(Cc, C, M, N, m_outer, n_outer);
      }
    }
  }
  free(Cc);
  free(Ac);
  free(Bc);
}

int main() {
  double *A, *B, *C, *CRef;
  int m, n, k, i, j;
  double alpha, beta;

  printf(
      "\n This example computes real matrix C=alpha*A*B+beta*C using \n"
      " Intel(R) MKL function dgemm, where A, B, and  C are matrices and \n"
      " alpha and beta are double precision scalars\n\n");

  m = 2000, k = 200, n = 1000;
  printf(
      " Initializing data for matrix multiplication C=A*B for matrix \n"
      " A(%ix%i) and matrix B(%ix%i)\n\n",
      m, k, k, n);
  alpha = 1.0;
  beta = 0.0;

  printf(
      " Allocating memory for matrices aligned on 64-byte boundary for "
      "better "
      "\n"
      " performance \n\n");
  A = (double *)malloc(m * k * sizeof(double));
  B = (double *)malloc(k * n * sizeof(double));
  C = (double *)malloc(m * n * sizeof(double));
  CRef = (double *)malloc(m * n * sizeof(double));

  if (A == NULL || B == NULL || C == NULL || CRef == NULL) {
    printf("\n ERROR: Can't allocate memory for matrices. Aborting... \n\n");
    free(A);
    free(B);
    free(C);
    free(CRef);
    return 1;
  }

  printf(" Intializing matrix data \n\n");
  for (i = 0; i < (m * k); i++) {
    A[i] = (double)(i + 1);
  }

  for (i = 0; i < (k * n); i++) {
    B[i] = (double)(-i - 1);
  }

  for (i = 0; i < (m * n); i++) {
    C[i] = 0.0;
    CRef[i] = 0.0;
  }

  set_math_flags();
  naive_dgemm(A, B, CRef, m, n, k);

  printf(
      " Computing matrix product using Intel(R) MKL dgemm function via CBLAS "
      "interface \n\n");
  manual_dgemm(A, B, C, m, n, k);

  for (i = 0; i < (m * n); i++) {
    if (abs(C[i] - CRef[i]) > 1e-4) {
      printf("error at %d\n", i);
      return 1;
    }
  }

  manual_dgemm(A, B, C, m, n, k);
  printf("\n Computations completed.\n\n");
  double elapsed = 1e6 * Halide::Tools::benchmark([&]() { manual_dgemm(A, B, C, m, n, k); });
  printf("time(us): %f, gflops: %f\n", elapsed, m * n * k * 2 * 1e-3 / elapsed);

  printf(" Top left corner of matrix A: \n");
  for (i = 0; i < std::min(m, 6); i++) {
    for (j = 0; j < std::min(k, 6); j++) {
      printf("%12.0f", A[j + i * k]);
    }
    printf("\n");
  }

  printf("\n Top left corner of matrix B: \n");
  for (i = 0; i < std::min(k, 6); i++) {
    for (j = 0; j < std::min(n, 6); j++) {
      printf("%12.0f", B[j + i * n]);
    }
    printf("\n");
  }

  printf("\n Top left corner of matrix C: \n");
  for (i = 0; i < std::min(m, 6); i++) {
    for (j = 0; j < std::min(n, 6); j++) {
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
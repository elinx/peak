#include <immintrin.h>

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

template <uint32_t M, uint32_t N, uint32_t K, uint32_t CN>
static inline void micro_kernel_intrincs_4x8(const double *A, const double *B, double *C) {
  double *C0 = C;
  double *C1 = C0 + CN;
  double *C2 = C1 + CN;
  double *C3 = C2 + CN;

  __m256d c0_0 = _mm256_load_pd(C0);
  __m256d c0_1 = _mm256_load_pd(&C0[4]);
  __m256d c1_0 = _mm256_load_pd(C1);
  __m256d c1_1 = _mm256_load_pd(&C1[4]);
  __m256d c2_0 = _mm256_load_pd(C2);
  __m256d c2_1 = _mm256_load_pd(&C2[4]);
  __m256d c3_0 = _mm256_load_pd(C3);
  __m256d c3_1 = _mm256_load_pd(&C3[4]);

  const double *Bp = B;
  const double *Ap = A;
  for (uint32_t k = 0; k < K; ++k) {
    // for (uint32_t m = 0; m < M; ++m) {  // M = 4
    // for (uint32_t n = 0; n < N; ++n) { // N = 8
    __m256d b0 = _mm256_load_pd(Bp);
    __m256d b1 = _mm256_load_pd(&Bp[4]);

    __m256d a0 = _mm256_broadcast_sd(&Ap[0]);
    __m256d a1 = _mm256_broadcast_sd(&Ap[1]);
    __m256d a2 = _mm256_broadcast_sd(&Ap[2]);
    __m256d a3 = _mm256_broadcast_sd(&Ap[3]);

    c0_0 = _mm256_fmadd_pd(a0, b0, c0_0);
    c0_1 = _mm256_fmadd_pd(a0, b1, c0_1);

    c1_0 = _mm256_fmadd_pd(a1, b0, c1_0);
    c1_1 = _mm256_fmadd_pd(a1, b1, c1_1);

    c2_0 = _mm256_fmadd_pd(a2, b0, c2_0);
    c2_1 = _mm256_fmadd_pd(a2, b1, c2_1);

    c3_0 = _mm256_fmadd_pd(a3, b0, c3_0);
    c3_1 = _mm256_fmadd_pd(a3, b1, c3_1);
    // }
    // }
    Bp += N;
    Ap += M;
  }

  _mm256_store_pd(C0, c0_0);
  _mm256_store_pd(&C0[4], c0_1);
  _mm256_store_pd(C1, c1_0);
  _mm256_store_pd(&C1[4], c1_1);
  _mm256_store_pd(C2, c2_0);
  _mm256_store_pd(&C2[4], c2_1);
  _mm256_store_pd(C3, c3_0);
  _mm256_store_pd(&C3[4], c3_1);
}

template <uint32_t M, uint32_t N, uint32_t K, uint32_t CN>
static inline void micro_kernel_intrincs(const double *A, const double *B, double *C) {
  double *C0 = C;
  double *C1 = C0 + CN;
  double *C2 = C1 + CN;
  double *C3 = C2 + CN;
  double *C4 = C3 + CN;
  double *C5 = C4 + CN;
  double *C6 = C5 + CN;
  double *C7 = C6 + CN;

  __m256d c0_0 = _mm256_load_pd(C0);
  __m256d c0_1 = _mm256_load_pd(&C0[4]);
  __m256d c1_0 = _mm256_load_pd(C1);
  __m256d c1_1 = _mm256_load_pd(&C1[4]);
  __m256d c2_0 = _mm256_load_pd(C2);
  __m256d c2_1 = _mm256_load_pd(&C2[4]);
  __m256d c3_0 = _mm256_load_pd(C3);
  __m256d c3_1 = _mm256_load_pd(&C3[4]);
  __m256d c4_0 = _mm256_load_pd(C4);
  __m256d c4_1 = _mm256_load_pd(&C4[4]);
  __m256d c5_0 = _mm256_load_pd(C5);
  __m256d c5_1 = _mm256_load_pd(&C5[4]);
  __m256d c6_0 = _mm256_load_pd(C6);
  __m256d c6_1 = _mm256_load_pd(&C6[4]);
  __m256d c7_0 = _mm256_load_pd(C7);
  __m256d c7_1 = _mm256_load_pd(&C7[4]);

  for (uint32_t k = 0; k < K; ++k) {
    // for (uint32_t m = 0; m < M; ++m) {  // M = 8
    // for (uint32_t n = 0; n < N; ++n) { // N = 8
    const double *Bp = &B[k * N];
    const double *Ap = &A[k * M];

    __m256d b0 = _mm256_load_pd(Bp);
    __m256d a1 = _mm256_broadcast_sd(&Ap[1]);
    __m256d b1 = _mm256_load_pd(&Bp[4]);
    __m256d a2 = _mm256_broadcast_sd(&Ap[2]);

    __m256d a0 = _mm256_broadcast_sd(&Ap[0]);

    c0_0 = _mm256_fmadd_pd(a0, b0, c0_0);
    __m256d a3 = _mm256_broadcast_sd(&Ap[3]);
    c0_1 = _mm256_fmadd_pd(a0, b1, c0_1);

    c1_0 = _mm256_fmadd_pd(a1, b0, c1_0);
    __m256d a4 = _mm256_broadcast_sd(&Ap[4]);
    c1_1 = _mm256_fmadd_pd(a1, b1, c1_1);

    c2_0 = _mm256_fmadd_pd(a2, b0, c2_0);
    __m256d a5 = _mm256_broadcast_sd(&Ap[5]);
    c2_1 = _mm256_fmadd_pd(a2, b1, c2_1);

    c3_0 = _mm256_fmadd_pd(a3, b0, c3_0);
    __m256d a6 = _mm256_broadcast_sd(&Ap[6]);
    c3_1 = _mm256_fmadd_pd(a3, b1, c3_1);

    c4_0 = _mm256_fmadd_pd(a4, b0, c4_0);
    __m256d a7 = _mm256_broadcast_sd(&Ap[7]);
    c4_1 = _mm256_fmadd_pd(a4, b1, c4_1);

    c5_0 = _mm256_fmadd_pd(a5, b0, c5_0);
    c5_1 = _mm256_fmadd_pd(a5, b1, c5_1);

    c6_0 = _mm256_fmadd_pd(a6, b0, c6_0);
    c6_1 = _mm256_fmadd_pd(a6, b1, c6_1);

    c7_0 = _mm256_fmadd_pd(a7, b0, c7_0);
    c7_1 = _mm256_fmadd_pd(a7, b1, c7_1);
    // C0[0] += Ap[0] * Bp[0];
    // C0[1] += Ap[0] * Bp[1];
    // C0[2] += Ap[0] * Bp[2];
    // C0[3] += Ap[0] * Bp[3];
    // C0[4] += Ap[0] * Bp[4];
    // C0[5] += Ap[0] * Bp[5];
    // C0[6] += Ap[0] * Bp[6];
    // C0[7] += Ap[0] * Bp[7];

    // C1[0] += Ap[1] * Bp[0];
    // C1[1] += Ap[1] * Bp[1];
    // C1[2] += Ap[1] * Bp[2];
    // C1[3] += Ap[1] * Bp[3];
    // C1[4] += Ap[1] * Bp[4];
    // C1[5] += Ap[1] * Bp[5];
    // C1[6] += Ap[1] * Bp[6];
    // C1[7] += Ap[1] * Bp[7];

    // C2[0] += Ap[2] * Bp[0];
    // C2[1] += Ap[2] * Bp[1];
    // C2[2] += Ap[2] * Bp[2];
    // C2[3] += Ap[2] * Bp[3];
    // C2[4] += Ap[2] * Bp[4];
    // C2[5] += Ap[2] * Bp[5];
    // C2[6] += Ap[2] * Bp[6];
    // C2[7] += Ap[2] * Bp[7];

    // C3[0] += Ap[3] * Bp[0];
    // C3[1] += Ap[3] * Bp[1];
    // C3[2] += Ap[3] * Bp[2];
    // C3[3] += Ap[3] * Bp[3];
    // C3[4] += Ap[3] * Bp[4];
    // C3[5] += Ap[3] * Bp[5];
    // C3[6] += Ap[3] * Bp[6];
    // C3[7] += Ap[3] * Bp[7];

    // C4[0] += Ap[4] * Bp[0];
    // C4[1] += Ap[4] * Bp[1];
    // C4[2] += Ap[4] * Bp[2];
    // C4[3] += Ap[4] * Bp[3];
    // C4[4] += Ap[4] * Bp[4];
    // C4[5] += Ap[4] * Bp[5];
    // C4[6] += Ap[4] * Bp[6];
    // C4[7] += Ap[4] * Bp[7];

    // C5[0] += Ap[5] * Bp[0];
    // C5[1] += Ap[5] * Bp[1];
    // C5[2] += Ap[5] * Bp[2];
    // C5[3] += Ap[5] * Bp[3];
    // C5[4] += Ap[5] * Bp[4];
    // C5[5] += Ap[5] * Bp[5];
    // C5[6] += Ap[5] * Bp[6];
    // C5[7] += Ap[5] * Bp[7];

    // C6[0] += Ap[6] * Bp[0];
    // C6[1] += Ap[6] * Bp[1];
    // C6[2] += Ap[6] * Bp[2];
    // C6[3] += Ap[6] * Bp[3];
    // C6[4] += Ap[6] * Bp[4];
    // C6[5] += Ap[6] * Bp[5];
    // C6[6] += Ap[6] * Bp[6];
    // C6[7] += Ap[6] * Bp[7];

    // C7[0] += Ap[7] * Bp[0];
    // C7[1] += Ap[7] * Bp[1];
    // C7[2] += Ap[7] * Bp[2];
    // C7[3] += Ap[7] * Bp[3];
    // C7[4] += Ap[7] * Bp[4];
    // C7[5] += Ap[7] * Bp[5];
    // C7[6] += Ap[7] * Bp[6];
    // C7[7] += Ap[7] * Bp[7];
    // }
    // }
  }

  _mm256_store_pd(C0, c0_0);
  _mm256_store_pd(&C0[4], c0_1);
  _mm256_store_pd(C1, c1_0);
  _mm256_store_pd(&C1[4], c1_1);
  _mm256_store_pd(C2, c2_0);
  _mm256_store_pd(&C2[4], c2_1);
  _mm256_store_pd(C3, c3_0);
  _mm256_store_pd(&C3[4], c3_1);
  _mm256_store_pd(C4, c4_0);
  _mm256_store_pd(&C4[4], c4_1);
  _mm256_store_pd(C5, c5_0);
  _mm256_store_pd(&C5[4], c5_1);
  _mm256_store_pd(C6, c6_0);
  _mm256_store_pd(&C6[4], c6_1);
  _mm256_store_pd(C7, c7_0);
  _mm256_store_pd(&C7[4], c7_1);
}

void manual_dgemm(const double *A, const double *B, double *C, const uint32_t M, const uint32_t N,
                  const uint32_t K) {
  constexpr uint32_t TILE_H = 4;
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
            micro_kernel_intrincs_4x8<m_inner_step, n_inner_step, k_inner_bound, n_inner_bound>(
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
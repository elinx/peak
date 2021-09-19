#include <immintrin.h>

#include <algorithm>
#include <cstdio>
#include <cstring>

#include "halide_benchmark.h"
#include "halide_macros.h"

constexpr uint32_t l1_cache_size = 192 * 1024;
constexpr uint32_t l2_cache_size = 1536 * 1024;
constexpr uint32_t l3_cache_size = 9 * 1024 * 1024;

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
  for (uint32_t nn = 0; nn < n_inner_bound; nn += n_inner_step) {
    for (uint32_t k = 0; k < k_inner_bound; ++k) {
      for (uint32_t n = 0; n < n_inner_step; ++n) {
        // TODO: remove this branch
        if ((k + k_outer) < K && (nn + n_outer + n) < N) {
          *Bp++ = B[k * N + n];
        } else {
          *Bp++ = 0;
        }
      }
    }
    B += n_inner_step;
  }
}

template <uint32_t m_inner_bound, uint32_t k_inner_bound, uint32_t m_inner_step>
static void PackA(double *Ap, const double *A, uint32_t M, uint32_t K, uint32_t m_outer,
                  uint32_t k_outer) {
  for (uint32_t mm = 0; mm < m_inner_bound; mm += m_inner_step) {
    for (uint32_t k = 0; k < k_inner_bound; ++k) {
      for (uint32_t m = 0; m < m_inner_step; ++m) {
        // TODO: remove this branch
        if ((k + k_outer) < K && (mm + m_outer + m) < M) {
          *Ap++ = A[m * K + k];
        } else {
          *Ap++ = 0;
        }
      }
    }
    A += m_inner_step * K;
  }
}

template <uint32_t m_inner_bound, uint32_t n_inner_bound, uint32_t m_inner_step,
          uint32_t n_inner_step>
static void WriteBackC(double *Cc, double *C, uint32_t M, uint32_t N, uint32_t m_outer,
                       uint32_t n_outer) {
  for (uint32_t nn = 0; nn < n_inner_bound; nn += n_inner_step) {
    for (uint32_t mm = 0; mm < m_inner_bound; mm += m_inner_step) {
      for (uint32_t m = 0; m < m_inner_step; ++m) {
        for (uint32_t n = 0; n < n_inner_step; ++n) {
          if ((m_outer + mm + m) < M && (n_outer + nn + n) < N) {
            C[(mm + m) * N + nn + n] += *Cc;
          }
          *Cc++ = 0;
        }
      }
    }
  }
}

template <uint32_t M, uint32_t N, uint32_t K>
static inline void micro_kernel(const double *A, const double *B, double *C) {
  for (uint32_t k = 0; k < K; ++k) {
    for (uint32_t m = 0; m < M; ++m) {
      for (uint32_t n = 0; n < N; ++n) {
        C[m * N + n] += A[k * M + m] * B[k * N + n];
      }
    }
  }
}

template <uint32_t M, uint32_t N, uint32_t K>
static inline void micro_kernel_intrincs_4x8(const double *A, const double *B, double *C) {
  double *C0 = C;
  double *C1 = C0 + N;
  double *C2 = C1 + N;
  double *C3 = C2 + N;

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
    __m256d b0 = _mm256_load_pd(Bp);
    __m256d b1 = _mm256_load_pd(Bp + 4);

    __m256d a0 = _mm256_broadcast_sd(Ap);
    __m256d a1 = _mm256_broadcast_sd(Ap + 1);
    __m256d a2 = _mm256_broadcast_sd(Ap + 2);
    __m256d a3 = _mm256_broadcast_sd(Ap + 3);

    c0_0 = _mm256_fmadd_pd(a0, b0, c0_0);
    c0_1 = _mm256_fmadd_pd(a0, b1, c0_1);

    c1_0 = _mm256_fmadd_pd(a1, b0, c1_0);
    c1_1 = _mm256_fmadd_pd(a1, b1, c1_1);

    c2_0 = _mm256_fmadd_pd(a2, b0, c2_0);
    c2_1 = _mm256_fmadd_pd(a2, b1, c2_1);

    c3_0 = _mm256_fmadd_pd(a3, b0, c3_0);
    c3_1 = _mm256_fmadd_pd(a3, b1, c3_1);
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

template <uint32_t M, uint32_t N, uint32_t K>
static inline void micro_kernel_intrincs_4x8_butterfly_permutation(const double *A, const double *B,
                                                                   double *C) {
  double *C0 = C;
  double *C1 = C0 + N;
  double *C2 = C1 + N;
  double *C3 = C2 + N;

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
    __m256d a0 = _mm256_load_pd(Ap);
    __m256d b0 = _mm256_load_pd(Bp);
    __m256d b1 = _mm256_load_pd(&Bp[4]);

    c0_0 = _mm256_fmadd_pd(a0, b0, c0_0);
    c0_1 = _mm256_fmadd_pd(a0, b1, c0_1);

    a0 = _mm256_permute4x64_pd(a0, _MM_SHUFFLE(2, 3, 0, 1));
    c1_0 = _mm256_fmadd_pd(a0, b0, c1_0);
    c1_1 = _mm256_fmadd_pd(a0, b1, c1_1);

    a0 = _mm256_permute2f128_pd(a0, a0, 0x03);
    c3_0 = _mm256_fmadd_pd(a0, b0, c3_0);
    c3_1 = _mm256_fmadd_pd(a0, b1, c3_1);

    a0 = _mm256_permute4x64_pd(a0, _MM_SHUFFLE(2, 3, 0, 1));
    c2_0 = _mm256_fmadd_pd(a0, b0, c2_0);
    c2_1 = _mm256_fmadd_pd(a0, b1, c2_1);
    Bp += N;
    Ap += M;
  }
  __m256d c0_0_semi = _mm256_shuffle_pd(c0_0, c1_0, 0b1010);
  __m256d c1_0_semi = _mm256_shuffle_pd(c1_0, c0_0, 0b1010);
  __m256d c2_0_semi = _mm256_shuffle_pd(c2_0, c3_0, 0b1010);
  __m256d c3_0_semi = _mm256_shuffle_pd(c3_0, c2_0, 0b1010);

  __m256d c0_1_semi = _mm256_shuffle_pd(c0_1, c1_1, 0b1010);
  __m256d c1_1_semi = _mm256_shuffle_pd(c1_1, c0_1, 0b1010);
  __m256d c2_1_semi = _mm256_shuffle_pd(c2_1, c3_1, 0b1010);
  __m256d c3_1_semi = _mm256_shuffle_pd(c3_1, c2_1, 0b1010);

  c0_0 = _mm256_permute2f128_pd(c0_0_semi, c2_0_semi, 0x30);
  c2_0 = _mm256_permute2f128_pd(c2_0_semi, c0_0_semi, 0x30);
  c1_0 = _mm256_permute2f128_pd(c1_0_semi, c3_0_semi, 0x30);
  c3_0 = _mm256_permute2f128_pd(c3_0_semi, c1_0_semi, 0x30);

  c0_1 = _mm256_permute2f128_pd(c0_1_semi, c2_1_semi, 0x30);
  c2_1 = _mm256_permute2f128_pd(c2_1_semi, c0_1_semi, 0x30);
  c1_1 = _mm256_permute2f128_pd(c1_1_semi, c3_1_semi, 0x30);
  c3_1 = _mm256_permute2f128_pd(c3_1_semi, c1_1_semi, 0x30);

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
  constexpr uint32_t TILE_K = 256;

  constexpr uint32_t m_outer_step = TILE_H * 64;
  constexpr uint32_t n_outer_step = TILE_W * 32;
  constexpr uint32_t k_outer_step = TILE_K;

  const uint32_t m_outer_bound = (M + m_outer_step - 1) / m_outer_step * m_outer_step;
  const uint32_t n_outer_bound = (N + n_outer_step - 1) / n_outer_step * n_outer_step;
  const uint32_t k_outer_bound = (K + n_outer_step - 1) / n_outer_step * n_outer_step;

  constexpr uint32_t m_inner_bound = m_outer_step;
  constexpr uint32_t n_inner_bound = n_outer_step;
  constexpr uint32_t k_inner_bound = k_outer_step;

  constexpr uint32_t m_inner_step = TILE_H;
  constexpr uint32_t n_inner_step = TILE_W;

  static_assert(!(n_inner_bound % n_inner_step));
  static_assert(!(m_inner_bound % m_inner_step));

  double *Bc = (double *)aligned_alloc(32, sizeof(double) * k_inner_bound * n_inner_bound);
  double *Ac = (double *)aligned_alloc(32, sizeof(double) * m_inner_bound * k_inner_bound);
  double *Cc = (double *)aligned_alloc(32, sizeof(double) * m_inner_bound * n_inner_bound);
  memset(C, 0, M * N * sizeof(double));

  for (uint32_t n_outer = 0; n_outer < n_outer_bound; n_outer += n_outer_step) {
    for (uint32_t k_outer = 0; k_outer < k_outer_bound; k_outer += k_outer_step) {
      PackB<k_inner_bound, n_inner_bound, n_inner_step>(Bc, &B[k_outer * N + n_outer], K, N,
                                                        k_outer, n_outer);
      for (uint32_t m_outer = 0; m_outer < m_outer_bound; m_outer += m_outer_step) {
        PackA<m_inner_bound, k_inner_bound, m_inner_step>(Ac, &A[m_outer * K + k_outer], M, K,
                                                          m_outer, k_outer);
        const double *Bcc = Bc;
        double *Ccc = Cc;
        for (uint32_t n_inner = 0; n_inner < n_inner_bound; n_inner += n_inner_step) {
          const double *Acc = Ac;
          for (uint32_t m_inner = 0; m_inner < m_inner_bound; m_inner += m_inner_step) {
            micro_kernel_intrincs_4x8_butterfly_permutation<m_inner_step, n_inner_step,
                                                            k_inner_bound>(Acc, Bcc, Ccc);
            Acc += m_inner_step * k_inner_bound;
            Ccc += m_inner_step * n_inner_step;
          }
          Bcc += n_inner_step * k_inner_bound;
        }
        WriteBackC<m_inner_bound, n_inner_bound, m_inner_step, n_inner_step>(
            Cc, &C[m_outer * N + n_outer], M, N, m_outer, n_outer);
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

  // m = 2000, k = 200, n = 1000;
  m = 640, k = 640, n = 640;
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
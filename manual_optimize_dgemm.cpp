#include <immintrin.h>

#include <algorithm>
#include <cstdio>
#include <cstring>

#include "halide_benchmark.h"
#include "halide_macros.h"

constexpr uint32_t l1_cache_size = 192 * 1024;
constexpr uint32_t l2_cache_size = 1536 * 1024;
constexpr uint32_t l3_cache_size = 9 * 1024 * 1024;

enum class MicroKernelType {
  kBroadcast,
  kButterflyPermunation,
};

enum class MicroKernelLang {
  kCpp,
  kIntrinsics,
  kAssembly,
};

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

// template <uint32_t k_inner_bound, uint32_t n_inner_bound, uint32_t n_inner_step>
// static void PackB(double *Bp, const double *B, uint32_t K, uint32_t N, uint32_t k_outer,
//                   uint32_t n_outer) {
//   for (uint32_t nn = 0; nn < n_inner_bound; nn += n_inner_step) {
//     for (uint32_t k = 0; k < k_inner_bound; ++k) {
//       for (uint32_t n = 0; n < n_inner_step; ++n) {
//         // TODO: remove this branch
//         if ((k + k_outer) < K && (nn + n_outer + n) < N) {
//           *Bp++ = B[k * N + n];
//         } else {
//           *Bp++ = 0;
//         }
//       }
//     }
//     B += n_inner_step;
//   }
// }

template <uint32_t k_inner_bound, uint32_t n_inner_bound, uint32_t n_inner_step>
static void PackB(double *Bp, const double *B, uint32_t K, uint32_t N, uint32_t k_outer,
                  uint32_t n_outer) {
  uint32_t k_size = std::min(k_inner_bound, K - k_outer);
  uint32_t n_size = std::min(n_inner_bound, N - n_outer);
  uint32_t k = 0;
  for (; k < k_size; ++k) {
    uint32_t n = 0;
    for (; n <= (n_size - n_inner_step); n += n_inner_step) {
      __m256d v1 = _mm256_load_pd(&B[k * N + n]);
      __m256d v2 = _mm256_load_pd(&B[k * N + n + 4]);
      _mm256_store_pd(&Bp[n * k_inner_bound + k * n_inner_step], v1);
      _mm256_store_pd(&Bp[n * k_inner_bound + k * n_inner_step + 4], v2);
      // for (uint32_t nn = 0; nn < n_inner_step; ++nn) {
      //   Bp[n * k_inner_bound + k * n_inner_step + nn] = B[k * N + n + nn];
      // }
    }
    for (; n < n_size; ++n) {
      uint32_t b = n / n_inner_step * n_inner_step * k_inner_bound;
      uint32_t s = k * n_inner_step;
      uint32_t p = n % n_inner_step;
      Bp[b + s + p] = B[k * N + n];
    }
    for (; n < n_inner_bound; ++n) {
      uint32_t b = n / n_inner_step * n_inner_step * k_inner_bound;
      uint32_t s = k * n_inner_step;
      uint32_t p = n % n_inner_step;
      Bp[b + s + p] = 0;
    }
  }
  memset((void *)&Bp[k * n_inner_bound], 0, (k_inner_bound - k) * n_inner_bound * sizeof(double));
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
  for (uint32_t m = 0; m < m_inner_step; ++m) {
    for (uint32_t n = 0; n < n_inner_step; ++n) {
      if ((m_outer + m) < M && (n_outer + n) < N) {
        C[(m)*N + n] += *Cc;
      }
      Cc++;
    }
  }
}

template <uint32_t M, uint32_t N, uint32_t K, MicroKernelType Type,
          MicroKernelLang Lang = MicroKernelLang::kCpp>
struct MicroKernel {
  static inline void run(const double *A, const double *B, double *C) {
    for (uint32_t k = 0; k < K; ++k) {
      for (uint32_t m = 0; m < M; ++m) {
        for (uint32_t n = 0; n < N; ++n) {
          C[m * N + n] += A[k * M + m] * B[k * N + n];
        }
      }
    }
  }
};

template <uint32_t K>
struct MicroKernel<4, 8, K, MicroKernelType::kBroadcast, MicroKernelLang::kIntrinsics> {
  static constexpr uint32_t M = 4;
  static constexpr uint32_t N = 8;
  static inline void run(const double *A, const double *B, double *C) {
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
};

template <uint32_t K>
struct MicroKernel<4, 8, K, MicroKernelType::kButterflyPermunation, MicroKernelLang::kIntrinsics> {
  static constexpr uint32_t M = 4;
  static constexpr uint32_t N = 8;
  static inline void run(const double *A, const double *B, double *C) {
    double *C0 = C;
    double *C1 = C0 + N;
    double *C2 = C1 + N;
    double *C3 = C2 + N;

    __m256d c0_0 = _mm256_setzero_pd();
    __m256d c0_1 = _mm256_setzero_pd();
    __m256d c1_0 = _mm256_setzero_pd();
    __m256d c1_1 = _mm256_setzero_pd();
    __m256d c2_0 = _mm256_setzero_pd();
    __m256d c2_1 = _mm256_setzero_pd();
    __m256d c3_0 = _mm256_setzero_pd();
    __m256d c3_1 = _mm256_setzero_pd();

    const double *Bp = B;
    const double *Ap = A;
    for (uint32_t k = 0; k < K; k += 4) {
      __m256d a0 = _mm256_load_pd(Ap);

      __m256d b0 = _mm256_load_pd(Bp);
      __m256d b1 = _mm256_load_pd(&Bp[4]);

      __m256d a1 = _mm256_permute4x64_pd(a0, _MM_SHUFFLE(2, 3, 0, 1));
      c0_0 = _mm256_fmadd_pd(a0, b0, c0_0);
      c0_1 = _mm256_fmadd_pd(a0, b1, c0_1);

      __m256d a2 = _mm256_permute2f128_pd(a1, a1, 0x03);
      c1_0 = _mm256_fmadd_pd(a1, b0, c1_0);
      c1_1 = _mm256_fmadd_pd(a1, b1, c1_1);

      __m256d a3 = _mm256_permute4x64_pd(a2, _MM_SHUFFLE(2, 3, 0, 1));
      c3_0 = _mm256_fmadd_pd(a2, b0, c3_0);
      c3_1 = _mm256_fmadd_pd(a2, b1, c3_1);

      c2_0 = _mm256_fmadd_pd(a3, b0, c2_0);
      c2_1 = _mm256_fmadd_pd(a3, b1, c2_1);

      a0 = _mm256_load_pd(Ap + M);

      b0 = _mm256_load_pd(Bp + N);
      b1 = _mm256_load_pd(&Bp[N + 4]);

      a1 = _mm256_permute4x64_pd(a0, _MM_SHUFFLE(2, 3, 0, 1));
      c0_0 = _mm256_fmadd_pd(a0, b0, c0_0);
      c0_1 = _mm256_fmadd_pd(a0, b1, c0_1);

      a2 = _mm256_permute2f128_pd(a1, a1, 0x03);
      c1_0 = _mm256_fmadd_pd(a1, b0, c1_0);
      c1_1 = _mm256_fmadd_pd(a1, b1, c1_1);

      a3 = _mm256_permute4x64_pd(a2, _MM_SHUFFLE(2, 3, 0, 1));
      c3_0 = _mm256_fmadd_pd(a2, b0, c3_0);
      c3_1 = _mm256_fmadd_pd(a2, b1, c3_1);

      c2_0 = _mm256_fmadd_pd(a3, b0, c2_0);
      c2_1 = _mm256_fmadd_pd(a3, b1, c2_1);

      a0 = _mm256_load_pd(Ap + 2 * M);

      b0 = _mm256_load_pd(Bp + 2 * N);
      b1 = _mm256_load_pd(&Bp[2 * N + 4]);

      a1 = _mm256_permute4x64_pd(a0, _MM_SHUFFLE(2, 3, 0, 1));
      c0_0 = _mm256_fmadd_pd(a0, b0, c0_0);
      c0_1 = _mm256_fmadd_pd(a0, b1, c0_1);

      a2 = _mm256_permute2f128_pd(a1, a1, 0x03);
      c1_0 = _mm256_fmadd_pd(a1, b0, c1_0);
      c1_1 = _mm256_fmadd_pd(a1, b1, c1_1);

      a3 = _mm256_permute4x64_pd(a2, _MM_SHUFFLE(2, 3, 0, 1));
      c3_0 = _mm256_fmadd_pd(a2, b0, c3_0);
      c3_1 = _mm256_fmadd_pd(a2, b1, c3_1);

      c2_0 = _mm256_fmadd_pd(a3, b0, c2_0);
      c2_1 = _mm256_fmadd_pd(a3, b1, c2_1);

      a0 = _mm256_load_pd(Ap + 3 * M);

      b0 = _mm256_load_pd(Bp + 3 * N);
      b1 = _mm256_load_pd(&Bp[3 * N + 4]);

      a1 = _mm256_permute4x64_pd(a0, _MM_SHUFFLE(2, 3, 0, 1));
      c0_0 = _mm256_fmadd_pd(a0, b0, c0_0);
      c0_1 = _mm256_fmadd_pd(a0, b1, c0_1);

      a2 = _mm256_permute2f128_pd(a1, a1, 0x03);
      c1_0 = _mm256_fmadd_pd(a1, b0, c1_0);
      c1_1 = _mm256_fmadd_pd(a1, b1, c1_1);

      a3 = _mm256_permute4x64_pd(a2, _MM_SHUFFLE(2, 3, 0, 1));
      c3_0 = _mm256_fmadd_pd(a2, b0, c3_0);
      c3_1 = _mm256_fmadd_pd(a2, b1, c3_1);

      c2_0 = _mm256_fmadd_pd(a3, b0, c2_0);
      c2_1 = _mm256_fmadd_pd(a3, b1, c2_1);

      Bp += 4 * N;
      Ap += 4 * M;
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
};

template <uint32_t K>
struct MicroKernel<4, 8, K, MicroKernelType::kButterflyPermunation, MicroKernelLang::kAssembly> {
  static constexpr uint32_t M = 4;
  static constexpr uint32_t N = 8;
  static inline void run(const double *A, const double *B, double *C) {
    uint64_t k_iter = K / 4;
    __asm__ volatile(
        "\n\t"
        "movq %0, %%rax\n\t"                    // A
        "movq %1, %%rbx\n\t"                    // B
        "movq %2, %%rcx\n\t"                    // C
        "vxorpd %%ymm8, %%ymm8, %%ymm8\n\t"     // c0_0
        "vxorpd %%ymm9, %%ymm9, %%ymm9\n\t"     // c0_1
        "vxorpd %%ymm10, %%ymm10, %%ymm10\n\t"  // c1_0
        "vxorpd %%ymm11, %%ymm11, %%ymm11\n\t"  // c1_1
        "vxorpd %%ymm12, %%ymm12, %%ymm12\n\t"  // c2_0
        "vxorpd %%ymm13, %%ymm13, %%ymm13\n\t"  // c2_1
        "vxorpd %%ymm14, %%ymm14, %%ymm14\n\t"  // c3_0
        "vxorpd %%ymm15, %%ymm15, %%ymm15\n\t"  // c3_1
        "movq %3, %%rsi\n\t"
        "vmovapd 0(%%rax), %%ymm0\n\t"    // [A] -> a0
        "vmovapd 0(%%rbx), %%ymm1\n\t"    // [B] -> b0
        "vmovapd 4*8(%%rbx), %%ymm2\n\t"  // [B+4] -> b1
        ".loop.start.L1:\n\t"
        "\n\t"
        "prefetcht0 192(%%rax)\n\t"
        "prefetcht0 384(%%rbx)\n\t"
        "vfmadd231pd %%ymm0, %%ymm1, %%ymm8\n\t"  // c0_0 += a0 * b0
        "vfmadd231pd %%ymm0, %%ymm2, %%ymm9\n\t"  // c0_1 += a0 * b1
        "vpermilpd $5, %%ymm0, %%ymm3\n\t"
        "vmovapd 32(%%rax), %%ymm4\n\t"            // [A+4] -> a0'
        "vfmadd231pd %%ymm3, %%ymm1, %%ymm10\n\t"  // c1_0 += a0 * b0
        "vfmadd231pd %%ymm3, %%ymm2, %%ymm11\n\t"  // c1_1 += a0 * b1
        "vperm2f128 $3, %%ymm3, %%ymm3, %%ymm0\n\t"
        "vmovapd 64(%%rbx), %%ymm5\n\t"            // [B+8] -> b0'
        "vfmadd231pd %%ymm0, %%ymm1, %%ymm14\n\t"  // c2_0 += a0 * b0
        "vfmadd231pd %%ymm0, %%ymm2, %%ymm15\n\t"  // c2_1 += a0 * b1
        "vpermilpd $5, %%ymm0, %%ymm3\n\t"
        "vmovapd 96(%%rbx), %%ymm6\n\t"            // [B+12] -> b1'
        "vfmadd231pd %%ymm3, %%ymm1, %%ymm12\n\t"  // c3_0 += a0 * b0
        "vfmadd231pd %%ymm3, %%ymm2, %%ymm13\n\t"  // c3_1 += a0 * b1
        "\n\t"
        "vfmadd231pd %%ymm4, %%ymm5, %%ymm8\n\t"  // c0_0 += a0 * b0
        "vfmadd231pd %%ymm4, %%ymm6, %%ymm9\n\t"  // c0_1 += a0 * b1
        "vpermilpd $5, %%ymm4, %%ymm7\n\t"
        "vmovapd 64(%%rax), %%ymm0\n\t"            // [A+8] -> a0
        "vfmadd231pd %%ymm7, %%ymm5, %%ymm10\n\t"  // c1_0 += a0 * b0
        "vfmadd231pd %%ymm7, %%ymm6, %%ymm11\n\t"  // c1_1 += a0 * b1
        "vperm2f128 $3, %%ymm7, %%ymm7, %%ymm4\n\t"
        "vmovapd 128(%%rbx), %%ymm1\n\t"           // [B+16] -> b0
        "vfmadd231pd %%ymm4, %%ymm5, %%ymm14\n\t"  // c2_0 += a0 * b0
        "vfmadd231pd %%ymm4, %%ymm6, %%ymm15\n\t"  // c2_1 += a0 * b1
        "vpermilpd $5, %%ymm4, %%ymm7\n\t"
        "vmovapd 160(%%rbx), %%ymm2\n\t"           // [B+20] -> b1
        "vfmadd231pd %%ymm7, %%ymm5, %%ymm12\n\t"  // c3_0 += a0 * b0
        "vfmadd231pd %%ymm7, %%ymm6, %%ymm13\n\t"  // c3_1 += a0 * b1
        "\n\t"
        "vfmadd231pd %%ymm0, %%ymm1, %%ymm8\n\t"  // c0_0 += a0 * b0
        "vfmadd231pd %%ymm0, %%ymm2, %%ymm9\n\t"  // c0_1 += a0 * b1
        "vpermilpd $5, %%ymm0, %%ymm3\n\t"
        "vmovapd 96(%%rax), %%ymm4\n\t"            // [A+4] -> a0'
        "vfmadd231pd %%ymm3, %%ymm1, %%ymm10\n\t"  // c1_0 += a0 * b0
        "vfmadd231pd %%ymm3, %%ymm2, %%ymm11\n\t"  // c1_1 += a0 * b1
        "vperm2f128 $3, %%ymm3, %%ymm3, %%ymm0\n\t"
        "vmovapd 192(%%rbx), %%ymm5\n\t"           // [B+8] -> b0'
        "vfmadd231pd %%ymm0, %%ymm1, %%ymm14\n\t"  // c2_0 += a0 * b0
        "vfmadd231pd %%ymm0, %%ymm2, %%ymm15\n\t"  // c2_1 += a0 * b1
        "vpermilpd $5, %%ymm0, %%ymm3\n\t"
        "vmovapd 224(%%rbx), %%ymm6\n\t"           // [B+12] -> b1'
        "vfmadd231pd %%ymm3, %%ymm1, %%ymm12\n\t"  // c3_0 += a0 * b0
        "vfmadd231pd %%ymm3, %%ymm2, %%ymm13\n\t"  // c3_1 += a0 * b1
        "\n\t"
        "vfmadd231pd %%ymm4, %%ymm5, %%ymm8\n\t"  // c0_0 += a0 * b0
        "vfmadd231pd %%ymm4, %%ymm6, %%ymm9\n\t"  // c0_1 += a0 * b1
        "vpermilpd $5, %%ymm4, %%ymm7\n\t"
        "vmovapd 128(%%rax), %%ymm0\n\t"           // [A+8] -> a0
        "vfmadd231pd %%ymm7, %%ymm5, %%ymm10\n\t"  // c1_0 += a0 * b0
        "vfmadd231pd %%ymm7, %%ymm6, %%ymm11\n\t"  // c1_1 += a0 * b1
        "vperm2f128 $3, %%ymm7, %%ymm7, %%ymm4\n\t"
        "vmovapd 256(%%rbx), %%ymm1\n\t"           // [B+16] -> b0
        "vfmadd231pd %%ymm4, %%ymm5, %%ymm14\n\t"  // c2_0 += a0 * b0
        "vfmadd231pd %%ymm4, %%ymm6, %%ymm15\n\t"  // c2_1 += a0 * b1
        "vpermilpd $5, %%ymm4, %%ymm7\n\t"
        "vmovapd 288(%%rbx), %%ymm2\n\t"           // [B+20] -> b1
        "vfmadd231pd %%ymm7, %%ymm5, %%ymm12\n\t"  // c3_0 += a0 * b0
        "vfmadd231pd %%ymm7, %%ymm6, %%ymm13\n\t"  // c3_1 += a0 * b1
        "\n\t"
        "addq $4*4*8, %%rax\n\t"  // A += 4*M
        "addq $4*8*8, %%rbx\n\t"  // B += 4*N
        "decq %%rsi\n\t"
        "jne .loop.start.L1"
        "\n\t"
        "vshufpd $10, %%ymm10, %%ymm8, %%ymm0\n\t"
        "vshufpd $10, %%ymm8, %%ymm10, %%ymm1\n\t"
        "vshufpd $10, %%ymm14, %%ymm12, %%ymm2\n\t"
        "vshufpd $10, %%ymm12, %%ymm14, %%ymm3\n\t"
        "vshufpd $10, %%ymm11, %%ymm9, %%ymm4\n\t"
        "vshufpd $10, %%ymm9, %%ymm11, %%ymm5\n\t"
        "vshufpd $10, %%ymm15, %%ymm13, %%ymm6\n\t"
        "vshufpd $10, %%ymm13, %%ymm15, %%ymm7\n\t"
        "\n\t"
        "vperm2f128 $48, %%ymm2, %%ymm0, %%ymm8\n\t"
        "vperm2f128 $48, %%ymm0, %%ymm2, %%ymm12\n\t"
        "vperm2f128 $48, %%ymm3, %%ymm1, %%ymm10\n\t"
        "vperm2f128 $48, %%ymm1, %%ymm3, %%ymm14\n\t"
        "vperm2f128 $48, %%ymm6, %%ymm4, %%ymm9\n\t"
        "vperm2f128 $48, %%ymm4, %%ymm6, %%ymm13\n\t"
        "vperm2f128 $48, %%ymm7, %%ymm5, %%ymm11\n\t"
        "vperm2f128 $48, %%ymm5, %%ymm7, %%ymm15\n\t"
        "\n\t"
        "vmovapd %%ymm8, (%%rcx)\n\t"
        "vmovapd %%ymm9, 32(%%rcx)\n\t"
        "vmovapd %%ymm10, 64(%%rcx)\n\t"
        "vmovapd %%ymm11, 96(%%rcx)\n\t"
        "vmovapd %%ymm12, 128(%%rcx)\n\t"
        "vmovapd %%ymm13, 160(%%rcx)\n\t"
        "vmovapd %%ymm14, 192(%%rcx)\n\t"
        "vmovapd %%ymm15, 224(%%rcx)\n\t"
        "\n\t"
        :            // output operands(none)
        :            // input operands
        "m"(A),      // 0
        "m"(B),      // 1
        "m"(C),      // 2
        "m"(k_iter)  // 3
        :            // register clobber list
        "rax", "rbx", "rcx", "rsi", "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6", "ymm7",
        "ymm8", "ymm9", "ymm10", "ymm11", "ymm12", "ymm13", "ymm14", "ymm15");
    // double *C0 = C;
    // double *C1 = C0 + N;
    // double *C2 = C1 + N;
    // double *C3 = C2 + N;

    // __m256d c0_0 = _mm256_setzero_pd();
    // __m256d c0_1 = _mm256_setzero_pd();
    // __m256d c1_0 = _mm256_setzero_pd();
    // __m256d c1_1 = _mm256_setzero_pd();
    // __m256d c2_0 = _mm256_setzero_pd();
    // __m256d c2_1 = _mm256_setzero_pd();
    // __m256d c3_0 = _mm256_setzero_pd();
    // __m256d c3_1 = _mm256_setzero_pd();

    // const double *Bp = B;
    // const double *Ap = A;
    // __m256d a0 = _mm256_load_pd(Ap);
    // __m256d b0 = _mm256_load_pd(Bp);
    // __m256d b1 = _mm256_load_pd(Bp + 4);

    // __m256d A0, B0, B1;
    // __m256d a1, a2, a3, A1, A2, A3;

    // for (uint32_t k = 0; k < K; k += 2) {
    //   __asm__ volatile("prefetcht0 192(%0)          \n\t" : : "r"(Bp));
    //   B0 = _mm256_load_pd(Bp + 8);

    //   c0_0 = _mm256_fmadd_pd(a0, b0, c0_0);
    //   c0_1 = _mm256_fmadd_pd(a0, b1, c0_1);

    //   B1 = _mm256_load_pd(Bp + 12);
    //   a1 = _mm256_permute4x64_pd(a0, _MM_SHUFFLE(2, 3, 0, 1));

    //   c1_0 = _mm256_fmadd_pd(a1, b0, c1_0);
    //   c1_1 = _mm256_fmadd_pd(a1, b1, c1_1);

    //   a2 = _mm256_permute2f128_pd(a1, a1, 0x03);
    //   A0 = _mm256_load_pd(Ap + 4);

    //   c3_0 = _mm256_fmadd_pd(a2, b0, c3_0);
    //   c3_1 = _mm256_fmadd_pd(a2, b1, c3_1);

    //   a3 = _mm256_permute4x64_pd(a2, _MM_SHUFFLE(2, 3, 0, 1));
    //   c2_0 = _mm256_fmadd_pd(a3, b0, c2_0);
    //   c2_1 = _mm256_fmadd_pd(a3, b1, c2_1);

    //   __asm__ volatile("prefetcht0 512(%0)          \n\t" : : "r"(Bp));

    //   b0 = _mm256_load_pd(Bp + 16);
    //   A1 = _mm256_permute4x64_pd(A0, _MM_SHUFFLE(2, 3, 0, 1));

    //   c0_0 = _mm256_fmadd_pd(A0, B0, c0_0);
    //   c0_1 = _mm256_fmadd_pd(A0, B1, c0_1);

    //   A2 = _mm256_permute2f128_pd(A1, A1, 0x03);

    //   c1_0 = _mm256_fmadd_pd(A1, B0, c1_0);
    //   c1_1 = _mm256_fmadd_pd(A1, B1, c1_1);

    //   b1 = _mm256_load_pd(Bp + 20);
    //   A3 = _mm256_permute4x64_pd(A2, _MM_SHUFFLE(2, 3, 0, 1));

    //   c3_0 = _mm256_fmadd_pd(A2, B0, c3_0);
    //   c3_1 = _mm256_fmadd_pd(A2, B1, c3_1);

    //   a0 = _mm256_load_pd(Ap + 8);

    //   c2_0 = _mm256_fmadd_pd(A3, B0, c2_0);
    //   c2_1 = _mm256_fmadd_pd(A3, B1, c2_1);

    //   Bp += 2 * N;
    //   Ap += 2 * M;
    // }

    // __m256d c0_0_semi = _mm256_shuffle_pd(c0_0, c1_0, 0b1010);
    // __m256d c1_0_semi = _mm256_shuffle_pd(c1_0, c0_0, 0b1010);
    // __m256d c2_0_semi = _mm256_shuffle_pd(c2_0, c3_0, 0b1010);
    // __m256d c3_0_semi = _mm256_shuffle_pd(c3_0, c2_0, 0b1010);

    // __m256d c0_1_semi = _mm256_shuffle_pd(c0_1, c1_1, 0b1010);
    // __m256d c1_1_semi = _mm256_shuffle_pd(c1_1, c0_1, 0b1010);
    // __m256d c2_1_semi = _mm256_shuffle_pd(c2_1, c3_1, 0b1010);
    // __m256d c3_1_semi = _mm256_shuffle_pd(c3_1, c2_1, 0b1010);

    // c0_0 = _mm256_permute2f128_pd(c0_0_semi, c2_0_semi, 0x30);
    // c2_0 = _mm256_permute2f128_pd(c2_0_semi, c0_0_semi, 0x30);
    // c1_0 = _mm256_permute2f128_pd(c1_0_semi, c3_0_semi, 0x30);
    // c3_0 = _mm256_permute2f128_pd(c3_0_semi, c1_0_semi, 0x30);

    // c0_1 = _mm256_permute2f128_pd(c0_1_semi, c2_1_semi, 0x30);
    // c2_1 = _mm256_permute2f128_pd(c2_1_semi, c0_1_semi, 0x30);
    // c1_1 = _mm256_permute2f128_pd(c1_1_semi, c3_1_semi, 0x30);
    // c3_1 = _mm256_permute2f128_pd(c3_1_semi, c1_1_semi, 0x30);

    // _mm256_store_pd(C0, c0_0);
    // _mm256_store_pd(&C0[4], c0_1);
    // _mm256_store_pd(C1, c1_0);
    // _mm256_store_pd(&C1[4], c1_1);
    // _mm256_store_pd(C2, c2_0);
    // _mm256_store_pd(&C2[4], c2_1);
    // _mm256_store_pd(C3, c3_0);
    // _mm256_store_pd(&C3[4], c3_1);
  }
};

template <uint32_t K>
struct MicroKernel<4, 12, K, MicroKernelType::kBroadcast, MicroKernelLang::kIntrinsics> {
  static constexpr uint32_t M = 4;
  static constexpr uint32_t N = 12;
  static inline void run(const double *A, const double *B, double *C) {
    double *C0 = C;
    double *C1 = C0 + N;
    double *C2 = C1 + N;
    double *C3 = C2 + N;

    __m256d c0_0 = _mm256_load_pd(C0);
    __m256d c0_1 = _mm256_load_pd(&C0[4]);
    __m256d c0_2 = _mm256_load_pd(&C0[8]);
    __m256d c1_0 = _mm256_load_pd(C1);
    __m256d c1_1 = _mm256_load_pd(&C1[4]);
    __m256d c1_2 = _mm256_load_pd(&C1[8]);
    __m256d c2_0 = _mm256_load_pd(C2);
    __m256d c2_1 = _mm256_load_pd(&C2[4]);
    __m256d c2_2 = _mm256_load_pd(&C2[8]);
    __m256d c3_0 = _mm256_load_pd(C3);
    __m256d c3_1 = _mm256_load_pd(&C3[4]);
    __m256d c3_2 = _mm256_load_pd(&C3[8]);

    const double *Bp = B;
    const double *Ap = A;
    for (uint32_t k = 0; k < K; ++k) {
      __m256d b0 = _mm256_load_pd(Bp);
      __m256d b1 = _mm256_load_pd(Bp + 4);
      __m256d b2 = _mm256_load_pd(Bp + 8);

      __m256d a0 = _mm256_broadcast_sd(Ap);

      c0_0 = _mm256_fmadd_pd(a0, b0, c0_0);
      c0_1 = _mm256_fmadd_pd(a0, b1, c0_1);
      c0_2 = _mm256_fmadd_pd(a0, b2, c0_2);

      a0 = _mm256_broadcast_sd(Ap + 1);
      c1_0 = _mm256_fmadd_pd(a0, b0, c1_0);
      c1_1 = _mm256_fmadd_pd(a0, b1, c1_1);
      c1_2 = _mm256_fmadd_pd(a0, b2, c1_2);

      a0 = _mm256_broadcast_sd(Ap + 2);
      c2_0 = _mm256_fmadd_pd(a0, b0, c2_0);
      c2_1 = _mm256_fmadd_pd(a0, b1, c2_1);
      c2_2 = _mm256_fmadd_pd(a0, b2, c2_2);

      a0 = _mm256_broadcast_sd(Ap + 3);
      c3_0 = _mm256_fmadd_pd(a0, b0, c3_0);
      c3_1 = _mm256_fmadd_pd(a0, b1, c3_1);
      c3_2 = _mm256_fmadd_pd(a0, b2, c3_2);
      Bp += N;
      Ap += M;
    }

    _mm256_store_pd(C0, c0_0);
    _mm256_store_pd(&C0[4], c0_1);
    _mm256_store_pd(&C0[8], c0_2);
    _mm256_store_pd(C1, c1_0);
    _mm256_store_pd(&C1[4], c1_1);
    _mm256_store_pd(&C1[8], c1_2);
    _mm256_store_pd(C2, c2_0);
    _mm256_store_pd(&C2[4], c2_1);
    _mm256_store_pd(&C2[8], c2_2);
    _mm256_store_pd(C3, c3_0);
    _mm256_store_pd(&C3[4], c3_1);
    _mm256_store_pd(&C3[8], c3_2);
  }
};
template <uint32_t K>
struct MicroKernel<3, 16, K, MicroKernelType::kBroadcast, MicroKernelLang::kIntrinsics> {
  static constexpr uint32_t M = 3;
  static constexpr uint32_t N = 16;
  static inline void run(const double *A, const double *B, double *C) {
    const double *Ap = A;
    const double *Bp = B;
    double *C0 = C;
    double *C1 = C + N;
    double *C2 = C1 + N;
    double a;

    __m256d c0_0 = _mm256_load_pd(C0);
    __m256d c0_1 = _mm256_load_pd(C0 + 4);
    __m256d c0_2 = _mm256_load_pd(C0 + 8);
    __m256d c0_3 = _mm256_load_pd(C0 + 12);

    __m256d c1_0 = _mm256_load_pd(C1);
    __m256d c1_1 = _mm256_load_pd(C1 + 4);
    __m256d c1_2 = _mm256_load_pd(C1 + 8);
    __m256d c1_3 = _mm256_load_pd(C1 + 12);

    __m256d c2_0 = _mm256_load_pd(C2);
    __m256d c2_1 = _mm256_load_pd(C2 + 4);
    __m256d c2_2 = _mm256_load_pd(C2 + 8);
    __m256d c2_3 = _mm256_load_pd(C2 + 12);

    for (uint32_t k = 0; k < K; ++k) {
      __m256d b0 = _mm256_load_pd(Bp);
      __m256d b1 = _mm256_load_pd(Bp + 4);
      __m256d b2 = _mm256_load_pd(Bp + 8);
      __m256d b3 = _mm256_load_pd(Bp + 12);

      __m256d a = _mm256_broadcast_sd(Ap);
      c0_0 = _mm256_fmadd_pd(a, b0, c0_0);
      c0_1 = _mm256_fmadd_pd(a, b1, c0_1);
      c0_2 = _mm256_fmadd_pd(a, b2, c0_2);
      c0_3 = _mm256_fmadd_pd(a, b3, c0_3);

      a = _mm256_broadcast_sd(Ap + 1);
      c1_0 = _mm256_fmadd_pd(a, b0, c1_0);
      c1_1 = _mm256_fmadd_pd(a, b1, c1_1);
      c1_2 = _mm256_fmadd_pd(a, b2, c1_2);
      c1_3 = _mm256_fmadd_pd(a, b3, c1_3);

      a = _mm256_broadcast_sd(Ap + 2);
      c2_0 = _mm256_fmadd_pd(a, b0, c2_0);
      c2_1 = _mm256_fmadd_pd(a, b1, c2_1);
      c2_2 = _mm256_fmadd_pd(a, b2, c2_2);
      c2_3 = _mm256_fmadd_pd(a, b3, c2_3);
      Ap += M;
      Bp += N;
    }

    _mm256_store_pd(C0, c0_0);
    _mm256_store_pd(C0 + 4, c0_1);
    _mm256_store_pd(C0 + 8, c0_2);
    _mm256_store_pd(C0 + 12, c0_3);

    _mm256_store_pd(C1, c1_0);
    _mm256_store_pd(C1 + 4, c1_1);
    _mm256_store_pd(C1 + 8, c1_2);
    _mm256_store_pd(C1 + 12, c1_3);

    _mm256_store_pd(C2, c2_0);
    _mm256_store_pd(C2 + 4, c2_1);
    _mm256_store_pd(C2 + 8, c2_2);
    _mm256_store_pd(C2 + 12, c2_3);
  }
};

template <uint32_t K>
struct MicroKernel<8, 8, K, MicroKernelType::kBroadcast, MicroKernelLang::kIntrinsics> {
  static constexpr uint32_t M = 8;
  static constexpr uint32_t N = 8;
  static inline void run(const double *A, const double *B, double *C) {
    double *C0 = C;
    double *C1 = C0 + N;
    double *C2 = C1 + N;
    double *C3 = C2 + N;
    double *C4 = C3 + N;
    double *C5 = C4 + N;
    double *C6 = C5 + N;
    double *C7 = C6 + N;

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
};

void manual_dgemm(const double *A, const double *B, double *C, const uint32_t M, const uint32_t N,
                  const uint32_t K) {
  constexpr uint32_t TILE_H = 4;
  constexpr uint32_t TILE_W = 8;
  constexpr uint32_t TILE_K = 320;

  constexpr uint32_t m_outer_step = TILE_H * 160;
  constexpr uint32_t n_outer_step = TILE_W * 4;
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
  static_assert(!(k_inner_bound % 2));

  double *Bc = (double *)aligned_alloc(32, sizeof(double) * k_inner_bound * n_inner_bound);
  double *Ac = (double *)aligned_alloc(32, sizeof(double) * m_inner_bound * k_inner_bound);
  double *Cc = (double *)aligned_alloc(32, sizeof(double) * m_inner_step * n_inner_step);
  memset(C, 0, M * N * sizeof(double));

  for (uint32_t k_outer = 0; k_outer < k_outer_bound; k_outer += k_outer_step) {
    for (uint32_t m_outer = 0; m_outer < m_outer_bound; m_outer += m_outer_step) {
      PackA<m_inner_bound, k_inner_bound, m_inner_step>(Ac, &A[m_outer * K + k_outer], M, K,
                                                        m_outer, k_outer);
      for (uint32_t n_outer = 0; n_outer < n_outer_bound; n_outer += n_outer_step) {
        PackB<k_inner_bound, n_inner_bound, n_inner_step>(Bc, &B[k_outer * N + n_outer], K, N,
                                                          k_outer, n_outer);
        for (uint32_t m_inner = 0; m_inner < m_inner_bound; m_inner += m_inner_step) {
          for (uint32_t n_inner = 0; n_inner < n_inner_bound; n_inner += n_inner_step) {
            MicroKernel<m_inner_step, n_inner_step, k_inner_bound,
                        MicroKernelType::kButterflyPermunation,
                        MicroKernelLang::kAssembly>::run(&Ac[m_inner * k_inner_bound],
                                                         &Bc[n_inner * k_inner_bound], Cc);
            WriteBackC<m_inner_bound, n_inner_bound, m_inner_step, n_inner_step>(
                Cc, &C[(m_outer + m_inner) * N + n_outer + n_inner], M, N, m_outer + m_inner,
                n_outer + n_inner);
          }
        }
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
  A = (double *)aligned_alloc(32, m * k * sizeof(double));
  B = (double *)aligned_alloc(32, k * n * sizeof(double));
  C = (double *)aligned_alloc(32, m * n * sizeof(double));
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

#if 1
  for (i = 0; i < (m * n); i++) {
    if (abs(C[i] - CRef[i]) > 1e-4) {
      printf("error at %d\n", i);
      return 1;
    }
  }

  uint32_t warmup = 100;
  while (warmup != 0) {
    manual_dgemm(A, B, C, m, n, k);
    warmup--;
  }
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
#endif

  printf("\n Deallocating memory \n\n");
  free(A);
  free(B);
  free(C);

  printf(" Example completed. \n\n");
  return 0;
}
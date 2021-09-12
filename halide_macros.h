/**
 * This macro is adapt from https://github.com/halide/Halide/blob/master/apps/linear_algebra/benchmarks/macros.h
 * Expaination: In Intel® processors, the flush-to-zero (FTZ) and denormals-are-zero (DAZ) flags in the
 * MXCSR register are used to control floating-point calculations. When the FTZ and DAZ flags are
 * enabled, the single instructions and multiple data (SIMD) floating-point computation can be
 * accelerated, thus improving the performance of the application.
 */

#define ENABLE_FTZ_DAZ 1
#ifdef ENABLE_FTZ_DAZ
#if defined(__i386__) || defined(__x86_64__)
#include <pmmintrin.h>
#include <xmmintrin.h>
#endif // defined(__i386__) || defined(__x86_64__)
#endif

inline void set_math_flags()
{
#ifdef ENABLE_FTZ_DAZ

#if defined(__i386__) || defined(__x86_64__)
    // Flush denormals to zero (the FTZ flag).
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    // Interpret denormal inputs as zero (the DAZ flag).
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif // defined(__i386__) || defined(__x86_64__)

#if defined(__arm__) || defined(__aarch64__)
    intptr_t fpsr = 0;

    // Get the FP status register
#if defined(__aarch64__)
    asm volatile("mrs %0, fpcr"
                 : "=r"(fpsr));
#else
    asm volatile("vmrs %0, fpscr"
                 : "=r"(fpsr));
#endif

    // Setting this is like setting FTZ+DAZ on x86
    constexpr intptr_t flush_to_zero = (1 << 24 /* FZ */);
    fpsr |= flush_to_zero;

    // Set the FP status register
#if defined(__aarch64__)
    asm volatile("msr fpcr, %0"
                 :
                 : "ri"(fpsr));
#else
    asm volatile("vmsr fpscr, %0"
                 :
                 : "ri"(fpsr));
#endif

#endif // defined(__arm__) || defined(__aarch64__)

#endif // ENABLE_FTZ_DAZ
}
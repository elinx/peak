# Introduction

This project helps to explore your hardware and achieve the peak performance for your needs.

# Enviroment

- OS: ubuntu 20.04
- Compiler: GCC 9.3.0
  - optimization level: -O3
- Platform: x86_x64
- Hardware info:

```
Architecture:                    x86_64
CPU op-mode(s):                  32-bit, 64-bit
Byte Order:                      Little Endian
Address sizes:                   39 bits physical, 48 bits virtual
CPU(s):                          6
On-line CPU(s) list:             0-5
Thread(s) per core:              1
Core(s) per socket:              6
Socket(s):                       1
Vendor ID:                       GenuineIntel
CPU family:                      6
Model:                           158
Model name:                      Intel(R) Core(TM) i5-8400 CPU @ 2.80GHz
Stepping:                        10
CPU MHz:                         2808.010
BogoMIPS:                        5616.02
Hypervisor vendor:               Microsoft
Virtualization type:             full
L1d cache:                       192 KiB
L1i cache:                       192 KiB
L2 cache:                        1.5 MiB
L3 cache:                        9 MiB
Vulnerability Itlb multihit:     KVM: Vulnerable
Vulnerability L1tf:              Mitigation; PTE Inversion
Vulnerability Mds:               Mitigation; Clear CPU buffers; SMT Host state unknown
Vulnerability Meltdown:          Mitigation; PTI
Vulnerability Spec store bypass: Mitigation; Speculative Store Bypass disabled via prctl and seccomp
Vulnerability Spectre v1:        Mitigation; usercopy/swapgs barriers and __user pointer sanitization
Vulnerability Spectre v2:        Mitigation; Full generic retpoline, IBPB conditional, IBRS_FW, STIBP disabled, RSB filling
Vulnerability Srbds:             Unknown: Dependent on hypervisor status
Vulnerability Tsx async abort:   Not affected
Flags:                           fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ss ht syscall nx pdpe1gb rdtscp lm constant_tsc rep_good nopl xt
                                 opology cpuid pni pclmulqdq ssse3 fma cx16 pcid sse4_1 sse4_2 movbe popcnt aes xsave avx f16c rdrand hypervisor lahf_lm abm 3dnowprefetch invpcid_single pti ssbd
                                 ibrs ibpb stibp fsgsbase bmi1 avx2 smep bmi2 erms invpcid rdseed adx smap clflushopt xsaveopt xsavec xgetbv1 xsaves md_clear flush_l1d arch_capabilities
```

- Miroarch: Coffee Lake (AVX2)

  - registers: 16 x 256bit
  - cache line size: 64bytes(8 FP64)

- Detailed Memory information

  | Level |    Type     |  Size  | Ways | Sets  | Latency |
  | :---: | :---------: | :----: | :--: | :---: | :-----: |
  |  L1   |    Data     | 32 KB  |  8   |  64   |
  |  L1   | Instruction | 32 KB  |  8   |  64   |
  |  L2   |    Data     | 256 KB |  4   | 1024  |
  |  L3   |    Data     |  9 MB  |  12  | 12288 |

- Instruction Timing

  |       Intrinsics        | Instruction | Latency | Throughput |
  | :---------------------: | :---------: | :-----: | :--------: |
  |     \_mm256_load_pd     |   vmovapd   |    7    |    0.5     |
  |    \_mm256_store_pd     |   vmovapd   |    5    |     1      |
  |    \_mm256_fmadd_pd     | vfmadd213pd |    4    |    0.5     |
  | \_mm256_permute4x64_pd  |   vpermpd   |    3    |     1      |
  | \_mm256_permute2f128_pd | vperm2f128  |    3    |     1      |
  |   \_mm256_shuffle_pd    |   vshufpd   |    1    |     1      |

- Coffee Lake microarch

![coffee lake microarch](./images/skylake_block_diagram.svg)

# Theorical Peak Performance

Coffee Lake microarch has two issue ports each has a `FMA` unit(see the previous diagram), so at most two floating point instructions(`add` and `mul`) could be done per-cycle, in addition the CPU supports `avx2` extension with 256bit register which could hold 4 FP64(8 FP32) values, as a result 16 FP64 instructions could be executed per-cycle. The CPU could boost it's frequency from 2.8GHz to at most 4GHz but we disable turbo mode for stablization, so the peak performance could be 44.8GFLOPS(FP64).

Also the following formula could be used:

```
N_CORES * FREQUENCY * FMA * UNITS * (SIZE_OF_VECTOR/64)
```

for my hardware in single thread

```
1 * 2.8 * 2 * 2 * (256/64) = 44.8
```

# Micro Kernel

$$
(m_rn_r + 2m_r + 2n_r ) × element\_size ≤ (n_f + n_{rf}) × p_f
$$

- $n_f$: number of floating point registers, 16 in this case
- $n_{rf}$:
- $p_f$: size of one floating point register in bytes, 32 in this case

# Memory Schedule

|               |  Shape   | Memory  | Memory(Readable) |
| :-----------: | :------: | :-----: | :--------------: |
| L1 Cache Size |          |  32768  |      32 KB       |
| L2 Cache Size |          | 262144  |      256 KB      |
| L3 Cache Size |          | 9437184 |       9 MB       |
|       M       |   640    |         |                  |
|       N       |   640    |         |                  |
|       K       |   640    |         |                  |
|      MR       |    4     |         |                  |
|      NR       |    8     |         |                  |
|      KC       |   256    |         |                  |
|      MC       |    64    |         |                  |
|      NC       |   640    |         |                  |
|       A       |  (M, K)  | 3276800 |     3.125 MB     |
|       B       |  (K, N)  | 3276800 |     3.125 MB     |
|       C       |  (M, N)  | 3276800 |     3.125 MB     |
|      Ac       | (MC, KC) | 131072  |      128 KB      |
|      Bc       | (KC, NC) | 1310720 |     1.25 MB      |
|   Ac-Slice    | (MR, KC) |  10240  |       8 KB       |
|   Bc-Slice    | (KC, NR) |  20480  |      16 KB       |
|      Cc       | (MR, NR) |   256   |      256 B       |

# Benchmarks

## Single Thread

|  Method   | GFLOPS(FP64) | Percentage |
| :-------: | :----------: | :--------: |
|  Theory   |     44.8     |     -      |
|    MKL    |  41.753118   |   93.20%   |
| OpenBlas  |  35.995601   |   80.34%   |
| naive-ijk |   1.455575   |   3.24%    |
|  manual   |  33.138241   |   73.97%   |

#if !defined(PADDLE_WITH_ARM) && !defined(PADDLE_WITH_SW) && \
    !defined(PADDLE_WITH_MIPS) && !defined(PADDLE_WITH_LOONGARCH)

#define __m256x __m256

static const unsigned int AVX_STEP_SIZE = 8;
static const unsigned int AVX_CUT_LEN_MASK = 7U;

#define _mm256_mul_px _mm256_mul_ps
#define _mm256_add_px _mm256_add_ps
#define _mm256_load_px _mm256_loadu_ps
#define _mm256_store_px _mm256_storeu_ps
#define _mm256_broadcast_sx _mm256_broadcast_ss

#define __m128x __m128

static const unsigned int SSE_STEP_SIZE = 2;
static const unsigned int SSE_CUT_LEN_MASK = 1U;

#define _mm_add_px _mm_add_ps
#define _mm_mul_px _mm_mul_ps
#define _mm_load_px _mm_loadu_ps
#define _mm_store_px _mm_storeu_ps
#define _mm_load1_px _mm_load1_ps



#if !defined(DOUBLE)

static const unsigned int RISCV_STEP_SIZE = 8;
static const unsigned int RISCV_CUT_LEN_MASK = 7U;

#define VSETVL(n)               __riscv_vsetvl_e32m8(n)
#define FLOAT_V_T               vfloat32m8_t
#define FLOAT_V_M1_T            vfloat32m1_t
#define VLEV_FLOAT              __riscv_vle32_v_f32m8
#define VLSEV_FLOAT             __riscv_vlse32_v_f32m8
#define VSEV_FLOAT              __riscv_vse32_v_f32m8
#define VSEV_FLOAT_M1           __riscv_vse32_v_f32m1
#define VSSEV_FLOAT             __riscv_vsse32_v_f32m8
#define VFMACCVF_FLOAT          __riscv_vfmacc_vf_f32m8
#define VFMVVF_FLOAT            __riscv_vfmv_v_f_f32m8
#define VFREDSUMVS_FLOAT        __riscv_vfredusum_vs_f32m8_f32m1
#define VFMVVF_FLOAT_M1         __riscv_vfmv_v_f_f32m1
#define VFMULVF_FLOAT            __rv__vfmul_vf_f32m8
#else

static const unsigned int RISCV_STEP_SIZE = 4;
static const unsigned int RISCV_CUT_LEN_MASK = 3U;

#define VSETVL(n)               __riscv_vsetvl_e64m8(n)
#define FLOAT_V_T               vfloat64m8_t
#define FLOAT_V_M1_T            vfloat64m1_t
#define VLEV_FLOAT              __riscv_vle64_v_f64m8
#define VLSEV_FLOAT             __riscv_vlse64_v_f64m8
#define VSEV_FLOAT              __riscv_vse64_v_f64m8
#define VSEV_FLOAT_M1           __riscv_vse64_v_f64m1
#define VSSEV_FLOAT             __riscv_vsse64_v_f64m8
#define VFMACCVF_FLOAT          __riscv_vfmacc_vf_f64m8
#define VFMVVF_FLOAT            __riscv_vfmv_v_f_f64m8
#define VFREDSUMVS_FLOAT        __riscv_vfredusum_vs_f64m8_f64m1
#define VFMVVF_FLOAT_M1         __riscv_vfmv_v_f_f64m1
#define VFMULVF_FLOAT            __rv__vfmul_vf_f64m8
#endif

#endif

template <typename T>
inline void axpy(const T* x, T* y, size_t len, const T alpha) {
  unsigned int jjj, lll;
  jjj = lll = 0;

#ifdef PADDLE_WITH_AVX
  lll = len & ~AVX_CUT_LEN_MASK;
  __m256x mm_alpha = _mm256_broadcast_sx(&alpha);    //将alpha这个数广播到向量mm_alpha中，是的这个向量的每一个元素都是alpha
  for (jjj = 0; jjj < lll; jjj += AVX_STEP_SIZE) {
    _mm256_store_px(
        y + jjj,
        _mm256_add_px(_mm256_load_px(y + jjj),
                      _mm256_mul_px(mm_alpha, _mm256_load_px(x + jjj))));
  }
#elif defined(PADDLE_WITH_ARM) || defined(PADDLE_WITH_SW) || \
    defined(PADDLE_WITH_MIPS) || defined(PADDLE_WITH_LOONGARCH)
  PADDLE_THROW(platform::errors::Unimplemented("axpy is not supported"));
#elif PADDLE_WITH_RISCV
  lll = len & ~RISCV_CUT_LEN_MASK;
  FLOAT_V_T mm_alpha = VFMVVF_FLOAT(alpha, VSETVL(RISCV_STEP_SIZE));
  for (jjj = 0; jjj < lll; jjj += RISCV_STEP_SIZE) {
    VSEV_FLOAT(
      y + jjj,
      VFMACCVF_FLOAT(mm_alpha, VLEV_FLOAT(x + jjj, VSETVL(RISCV_STEP_SIZE)), VLEV_FLOAT(y + jjj, VSETVL(RISCV_STEP_SIZE)), VSETVL(RISCV_STEP_SIZE)));
}
#else
  lll = len & ~SSE_CUT_LEN_MASK;
  __m128x mm_alpha = _mm_load1_px(&alpha);
  for (jjj = 0; jjj < lll; jjj += SSE_STEP_SIZE) {
    _mm_store_px(y + jjj,
                 _mm_add_px(_mm_load_px(y + jjj),
                            _mm_mul_px(mm_alpha, _mm_load_px(x + jjj))));
  }

#endif

  for (; jjj < len; jjj++) {
    y[jjj] += alpha * x[jjj];
  }
}


template <typename T>
inline void axpy_noadd(const T* x, T* y, size_t len, const T alpha) {
  unsigned int jjj, lll;
  jjj = lll = 0;

#ifdef PADDLE_WITH_AVX
  lll = len & ~AVX_CUT_LEN_MASK;
  __m256x mm_alpha = _mm256_broadcast_sx(&alpha);
  for (jjj = 0; jjj < lll; jjj += AVX_STEP_SIZE) {
    _mm256_store_px(y + jjj, _mm256_mul_px(mm_alpha, _mm256_load_px(x + jjj)));
  }
#elif defined(PADDLE_WITH_ARM) || defined(PADDLE_WITH_SW) || \
    defined(PADDLE_WITH_MIPS) || defined(PADDLE_WITH_LOONGARCH)
  PADDLE_THROW(platform::errors::Unimplemented("axpy_noadd is not supported"));
#elif PADDLE_WITH_RISCV
  lll = len & ~RISCV_CUT_LEN_MASK;
  FLOAT_V_T mm_alpha = VFMVVF_FLOAT(alpha, VSETVL(RISCV_STEP_SIZE));
  for (jjj = 0; jjj < lll; jjj += RISCV_STEP_SIZE) {
    VSEV_FLOAT(
      y + jjj,
      VFMULVF_FLOAT(mm_alpha, VLEV_FLOAT(x + jjj, VSETVL(RISCV_STEP_SIZE)), VSETVL(RISCV_STEP_SIZE)));
  }
#else
  lll = len & ~SSE_CUT_LEN_MASK;
  __m128x mm_alpha = _mm_load1_px(&alpha);
  for (jjj = 0; jjj < lll; jjj += SSE_STEP_SIZE) {
    _mm_store_px(y + jjj, _mm_mul_px(mm_alpha, _mm_load_px(x + jjj)));
  }

#endif

  for (; jjj < len; jjj++) {
    y[jjj] = alpha * x[jjj];
  }
}
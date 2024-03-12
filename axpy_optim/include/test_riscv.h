#include "common.h"

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

template <typename T>
inline void axpy_riscv(const T* x, T* y, size_t len, const T alpha) {
  unsigned int jjj, lll;
  jjj = lll = 0;

  lll = len & ~RISCV_CUT_LEN_MASK;
  FLOAT_V_T mm_alpha = VFMVVF_FLOAT(alpha, VSETVL(RISCV_STEP_SIZE));
  for (jjj = 0; jjj < lll; jjj += RISCV_STEP_SIZE) {
    VSEV_FLOAT(
      y + jjj,
      VFMACCVF_FLOAT(mm_alpha, VLEV_FLOAT(x + jjj, VSETVL(RISCV_STEP_SIZE)), VLEV_FLOAT(y + jjj, VSETVL(RISCV_STEP_SIZE)), VSETVL(RISCV_STEP_SIZE)));
  }

  for (; jjj < len; jjj++) {
    y[jjj] += alpha * x[jjj];
  }
}


template <typename T>
inline void axpy_noadd_riscv(const T* x, T* y, size_t len, const T alpha) {
  unsigned int jjj, lll;
  jjj = lll = 0;

  lll = len & ~RISCV_CUT_LEN_MASK;
  FLOAT_V_T mm_alpha = VFMVVF_FLOAT(alpha, VSETVL(RISCV_STEP_SIZE));
  for (jjj = 0; jjj < lll; jjj += RISCV_STEP_SIZE) {
    VSEV_FLOAT(
      y + jjj,
      VFMULVF_FLOAT(mm_alpha, VLEV_FLOAT(x + jjj, VSETVL(RISCV_STEP_SIZE)), VSETVL(RISCV_STEP_SIZE)));
  }

  for (; jjj < len; jjj++) {
    y[jjj] = alpha * x[jjj];
  }
}
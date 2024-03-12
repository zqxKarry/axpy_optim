#include <immintrin.h>
#include <cfloat>
#include <cmath>
#include <cstring>

#define __m256x __m256

static const unsigned int AVX_STEP_SIZE = 8;
static const unsigned int AVX_CUT_LEN_MASK = 7U;

#define _mm256_mul_px _mm256_mul_ps
#define _mm256_add_px _mm256_add_ps
#define _mm256_load_px _mm256_loadu_ps
#define _mm256_store_px _mm256_storeu_ps
#define _mm256_broadcast_sx _mm256_broadcast_ss

// #define __m128x __m128

// static const unsigned int SSE_STEP_SIZE = 2;
// static const unsigned int SSE_CUT_LEN_MASK = 1U;

// #define _mm_add_px _mm_add_ps
// #define _mm_mul_px _mm_mul_ps
// #define _mm_load_px _mm_loadu_ps
// #define _mm_store_px _mm_storeu_ps
// #define _mm_load1_px _mm_load1_ps

template <typename T>
inline void axpy_avx(const T* x, T* y, size_t len, const T alpha) {
  unsigned int jjj, lll;
  jjj = lll = 0;

  lll = len & ~AVX_CUT_LEN_MASK;
  __m256x mm_alpha = _mm256_broadcast_sx(&alpha);    //将alpha这个数广播到向量mm_alpha中，是的这个向量的每一个元素都是alpha
  for (jjj = 0; jjj < lll; jjj += AVX_STEP_SIZE) {
    _mm256_store_px(
        y + jjj,
        _mm256_add_px(_mm256_load_px(y + jjj),
                      _mm256_mul_px(mm_alpha, _mm256_load_px(x + jjj))));
  }
  for (; jjj < len; jjj++) {
    y[jjj] += alpha * x[jjj];
  }
}


template <typename T>
inline void axpy_noadd_avx(const T* x, T* y, size_t len, const T alpha) {
  unsigned int jjj, lll;
  jjj = lll = 0;

  lll = len & ~AVX_CUT_LEN_MASK;
  __m256x mm_alpha = _mm256_broadcast_sx(&alpha);
  for (jjj = 0; jjj < lll; jjj += AVX_STEP_SIZE) {
    _mm256_store_px(y + jjj, _mm256_mul_px(mm_alpha, _mm256_load_px(x + jjj)));
  }
  
  for (; jjj < len; jjj++) {
    y[jjj] = alpha * x[jjj];
  }
}
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
#define HAS_TMA 1
#else
#define HAS_TMA 0
#endif

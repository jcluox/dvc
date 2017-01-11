#ifndef LDPCA_CUDA_H
#define LDPCA_CUDA_H

#include "cutil.h"
#include "defines.h"
#include <cuda.h>     //CUDA Driver API

#ifdef __cplusplus 
extern "C" { 
#endif

extern int cuda_concurrent_copy_execution;
extern float *d_LR_intrinsic;
extern float *d_entropy;

bool InitCUDA();
void LDPCA_CUDA_Init_Variable();
void LDPCA_CUDA_Free_Variable();
int beliefPropagation_CUDA(unsigned char *syndrome, unsigned char *decoded, int nCode, int request);
int beliefPropagation_CUDA_earlyJump(unsigned char *syndrome, unsigned char *decoded, int nCode, int request);
float ComputeCE_FERMI();

#ifdef __cplusplus 
}
#endif

#endif
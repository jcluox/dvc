#ifndef NOISEMODEL_KERNEL_H
#define NOISEMODEL_KERNEL_H

#include "cutil.h"
#include "global.h"

extern SMF *d_smf;
extern SearchInfo *d_searchInfo;

#ifdef __cplusplus 
extern "C" { 
#endif
void CNM_CUDA_Init_Buffer();
void CNM_CUDA_Free_Buffer();
void CNM_CUDA_Init_Variable(SIR *sir, SMF *smf);
void init_noiseDst_CUDA(int range, int indexInBlk, int searchCandidate);
void free_noiseDst_CUDA();
void noiseDistribution_CUDA(int range, int indexInBlk, float *alpha, SMF *smf, Trans *overcomplete_trans, SIR *sir);
void condBitProb_CUDA(float *LLR_intrinsic, int indexInBlk, int bitOfCoeff, int stepSize, int *reconsTransFrame, int range, SIR *sir);
#ifdef __cplusplus 
}
#endif
#endif
#ifndef MOTIONLEARNING_KERNEL_H
#define MOTIONLEARNING_KERNEL_H

#include "cutil.h"
#include "global.h"


#ifdef __cplusplus 
extern "C" { 
#endif
void updateSMF_CUDA_Init_Buffer();
void updateSMF_CUDA_Free_Buffer();
void updateSMF_CUDA(SMF *smf, unsigned char *reconsFrame, unsigned char  *sideInfoFrame, SIR *sir);
#ifdef __cplusplus 
}
#endif
#endif
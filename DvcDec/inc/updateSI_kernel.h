#ifndef UPDATESI_KERNEL_H
#define UPDATESI_KERNEL_H

#include "cutil.h"
#include "global.h"

extern unsigned char *d_reconsFrame;
extern unsigned char  *d_sideInfoFrame;

#ifdef __cplusplus 
extern "C" { 
#endif
void ME_CUDA_Init_Buffer();
void ME_CUDA_Free_Buffer();
void ME_CUDA_Init_Variable(SIR *sir, int begin, int end);
//void ME_CUDA_Free_Variable();
void updateSI_CUDA(SideInfo *si, unsigned char *reconsFrame, int current, int past, int future, SIR *sir);
#ifdef __cplusplus 
}
#endif
#endif
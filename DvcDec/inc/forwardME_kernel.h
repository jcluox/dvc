#ifndef FORWARDME_KERNEL_H
#define FORWARDME_KERNEL_H

#include "cutil.h"
#include "global.h"


#ifdef __cplusplus 
extern "C" { 
#endif
void ForwardME_CUDA_Init_Buffer();
void ForwardME_CUDA_Free_Buffer();
void forwardME_CUDA(int future, int past, int current, SideInfo *si);
#ifdef __cplusplus 
}
#endif
#endif
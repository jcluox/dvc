#ifndef NOISEMODEL_H
#define NOISEMODEL_H

#include "global.h"

void noiseParameter(int *transFrame, float *alpha, unsigned char *BlockModeFlag);
float *noiseDistribution(int range, int indexInBlk, float *alpha, SMF *smf, Trans *overcomplete_trans, SIR *sir);
float *noiseDistribution_OpenMP(int range, int indexInBlk, float *alpha, SMF *smf, Trans *overcomplete_trans, SIR *sir);
void condBitProb(float *LLR_intrinsic, int indexInBlk, int bitOfCoeff, int stepSize, int *reconsTransFrame, int range, float *distribution, SIR *sir);
void condBitProb_OpenMP(float *LLR_intrinsic, int indexInBlk, int bitOfCoeff, int stepSize, int *reconsTransFrame, int range, float *distribution, SIR *sir);

#endif


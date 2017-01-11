#ifndef RECONSTRUCTION_H
#define RECONSTRUCTION_H

#include "global.h"

extern int *Qtable;
extern int *QtableBits;
extern int QbitsNumPerBlk;	//amount of bitplanes per frame
extern int ACLevelShift[BLOCKSIZE44];	//AC level shift (let all coefficients >= 0)
extern int QuantizationTable[9][BLOCKSIZE44];
extern int QuantizationTableBits[9][BLOCKSIZE44];

void quantize44(int *blk, int *stepSize);
void reconstruction(int *reconsTransFrame, int *sideTransFrame, int indexInblk, float *alpha, int *stepSize, unsigned char *BlockModeFlag, unsigned char *deblockFlag);
void reconstruction_OpenMP(int *reconsTransFrame, int *sideTransFrame, int indexInblk, float *alpha, int *stepSize, unsigned char *BlockModeFlag, unsigned char *deblockFlag);

#endif
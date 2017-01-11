#ifndef QUANTIZATION_H
#define QUANTIZATION_H

#include "defines.h"

//use for quantization
extern int *Qtable;
extern int *QtableBits;
extern int QbitsNumPerBlk;	//number of bits per block
extern int ACLevelShift[BLOCKSIZE44];	//AC level shift (let all coefficients >= 0)
extern int stepSize[BLOCKSIZE44];
extern int QuantizationTable[9][BLOCKSIZE44];
extern int QuantizationTableBits[9][BLOCKSIZE44];

void quantize44(int *blk, int *stepSize);
//perform quantization to transFrame
void quantize(int *transFrame, EncodedFrame *encodedFrame);

#endif
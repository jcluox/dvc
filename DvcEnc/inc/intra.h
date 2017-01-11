#ifndef INTRA_H
#define INTRA_H

#include "global.h"

int intra_prediction(int frameIdx,int X,int Y,EncodedFrame *encodedFrame,int block44pos,int qbits,int offset,double Qstep,int write_flag,int recal_intra);
void real_intra(EncodedFrame *encodedFrame, int frameIdx);

#endif
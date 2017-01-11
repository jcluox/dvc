#ifndef TRANSFORM_H
#define TRANSFORM_H

#include "global.h"

extern double *Kmatrix;
extern double postScale[BLOCKSIZE44];
extern double preScale[BLOCKSIZE44];
//for H264 Intra
extern int MF[BLOCKSIZE44];
extern int *qp_rem_matrix;
extern int *qp_per_matrix;

void initTransform();
void forward44(int *blk);
void inverse44(int *blk);
void forwardTransform(unsigned char *oriFrame, int *transFrame, EncodedFrame *encodedFrame);
void forwardTransform_Pure(unsigned char *oriFrame, int *transFrame, EncodedFrame *encodedFrame);
void inverseTransform(int *transFrame, unsigned char *refFrame, unsigned char *reconsFrame, EncodedFrame *encodedFrame);

#endif
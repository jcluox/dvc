#ifndef TRANSFORM_H
#define TRANSFORM_H

#include "global.h"

extern double postScale[BLOCKSIZE44];
extern double preScale[BLOCKSIZE44];
//for intra block
extern int *qp_per_matrix;
extern int *qp_rem_matrix;

void initTransform();
void forward44(int *blk);
void inverse44(int *blk);
void forwardTransform(int *oriFrame, int *transFrame);
void inverseTransform(int *transFrame, unsigned char *reconsFrame, SIR *sir);
void inverseTransform_OpenMP(int *transFrame, unsigned char *reconsFrame, SIR *sir);
void overCompleteTransform(unsigned char *pixelFrame, Trans *transFrame, int searchRangeBottom, int searchRangeRight);
void overCompleteTransform_OpenMP(unsigned char *pixelFrame, Trans *transFrame, int searchRangeBottom, int searchRangeRight);

#endif
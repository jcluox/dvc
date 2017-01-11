#ifndef FILTER_H
#define FILTER_H

extern double FilterCoef[3][SQR_FILTER];

void FIR_filter(unsigned char *inFrame, int width, int height, unsigned char *outFrame);
void FIR_filter_OpenMP(unsigned char *inFrame, int width, int height, unsigned char *outFrame);
void bilinear_filter(unsigned char *inFrame, int width, int height, unsigned char *outFrame);
void lowPassFilter(unsigned char *inFrame, int width, int height, unsigned char *outFrame);

#endif
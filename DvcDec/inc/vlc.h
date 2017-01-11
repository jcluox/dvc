#ifndef VLC_H
#define VLC_H

#include "global.h"

void readCoeff4x4_CAVLC(int X, int Y, int levarr[BLOCKSIZE44], int runarr[BLOCKSIZE44],int *number_coefficients, EncodedFrame *encodedFrame);

#endif
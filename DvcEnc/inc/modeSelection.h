#ifndef MODESELECTION_H
#define MODESELECTION_H

#include "global.h"

typedef struct intra_param{
    double thita;
    double kappa;
}Intra_Param;
extern Intra_Param intrap;
void recal_decision(EncodedFrame *encodedFrame);
void intra_selection(EncodedFrame *encodedFrame, unsigned char *sideInfoFrame, int frameIdx);

#endif
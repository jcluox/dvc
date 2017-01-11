#ifndef MOTIONLEARNING_H
#define MOTIONLEARNING_H

#include "global.h"

void initSMF(SMF *smf, SIR *sir);
void init_si_ML(int *si_ML, Trans *transFrame);
void updateSMF(SMF *smf, unsigned char *reconsFrame, unsigned char *sideInfoFrame, SIR *sir);
void update_si_ML(int *sideTransFrame, Trans *transFrame, int band, SMF *smf, SIR *sir);
void updateSMF_OpenMP(SMF *smf, unsigned char *reconsFrame, unsigned char *sideInfoFrame, SIR *sir);
void update_si_ML_OpenMP(int *sideTransFrame, Trans *transFrame, int band, SMF *smf, SIR *sir);

#endif

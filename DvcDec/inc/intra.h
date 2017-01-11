#ifndef INTRA_H
#define INTRA_H

#include "global.h"

void intraRecon(unsigned char *reconsFrame, EncodedFrame *encodedFrame);
void copyIntraBlkToSideInfo(unsigned char *sideInfoFrame, unsigned char *reconsFrame, unsigned char *BlockModeFlag);

#endif
#ifndef BITPLANE_H
#define BITPLANE_H

void putBitplaneToCoeff(unsigned char *decodedBitplane, int *reconsTransFrame, int indexInBlk, unsigned char *BlockModeFlag);
void copyNonSendCeff(int *reconsTransFrame, int *sideTransFrame, unsigned char *BlockModeFlag);
void copySkipBlk(unsigned char *reconsFrame, unsigned char *refFrame, unsigned char *BlockModeFlag);

#endif
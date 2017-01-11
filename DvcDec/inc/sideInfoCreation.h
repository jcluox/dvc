#ifndef SIDEINFOCREATION
#define SIDEINFOCREATION

#include "global.h"

double costFunc(double mad, Point *mv);
int computeEndPoint(double stPoint, double vector, double scale, int lower, int upper);
void init_1616ME(SearchInfo *searchInfo1616);
void forwardME(int future, int past, int current, SideInfo *si);
void bidirectME(int future, int past, int current, MV *mv, int blockLen);
void motionFilterAndComp(int future, int past, int current, SideInfo *si);
void createSideInfo(int past, int future, int current, SideInfo *si);

#if 1	//search each integer pixel -> 8 half pixel around it

void initSIR(SIR *sir, unsigned char *BlockModeFlag);
void subPelRefine(double *block, unsigned char *referenceFrame, int *bestX, int *bestY, int width2, int height2);
void updateSI(SideInfo *si, unsigned char *reconsFrame, int current, int past, int future, SIR *sir);
void updateSI_OpenMP(SideInfo *si, unsigned char *reconsFrame, int current, int past, int future, SIR *sir);
//void updateSI_CUDA(SideInfo *si, unsigned char *reconsFrame, int current, int past, int future, SIR *sir);
#else	//search each half pixel

void initSIR(SIR *sir, unsigned char *BlockModeFlag);
void updateSI(SideInfo *si, unsigned char *reconsFrame, int current, int past, int future, SIR *sir);
void updateSI_OpenMP(SideInfo *si, unsigned char *reconsFrame, int current, int past, int future, SIR *sir);

#endif

#endif
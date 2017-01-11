#include "global.h"
#include <stdlib.h>

CodingOption codingOption;

const char *parameters[PARAMETERNUM] = {
	"OriginalSequence",			//original yuv (used for PSNR calculation)
	"WZBitstream",				//WZ bitstream file
	"KeyFrameSequence",			//reconstruct yuv by h264
	"KeyFrameInfo",				//key frame information output by h264
	"RDFile",					//report decoding result
	"OutputSequences",			//decoded yuv
    "FrameRate",				//used for rate calculation
	"ParallelMode",				//0: sequential, 1: OpenMP only, 2: CUDA only, 3: OpenMP + CUDA
	"UseCoreNum",				//number of CPU used by OpenMP (ParallelMode: 1 or 3)
	"IntraBitstream"			//Intra block bitstream file
};

int FrameSize;
int Block44Num;
int Block1616Num;	//macroblock number
int Block88Num;
int Block44Height, Block44Width;
int Block88Height, Block88Width;
int Block1616Height, Block1616Width;
unsigned char **oriFrames;  //original sequence (only y) used for PSNR calculation

EncodedFrame *encodedFrames;

DecodedFrame *decodedFrames;

SideInfo sideInfo;

SearchInfo *SearchInfo1616;

int zigzag[BLOCKSIZE44] = {
	0, 1, 4, 8,
	5, 2, 3, 6,
	9,12,13,10,
	7,11,14,15,
};

double CPUFreq;

#ifdef SHOW_TIME_INFO
double Total_overT_time=0.0;
double Total_ldpca_time=0.0;
double Total_CNM_time=0.0;
double Total_condBit_time=0.0;
double Total_updateSI_time=0.0;
double Total_sideCreate_time=0.0;
double Total_ML_time=0.0;
#endif
//double byte2bitTime=0.0;
/*add by JSRF*/
#ifdef LDPC_ANALYSIS
LdpcInfo *ldpcInfo;
#endif

void timeStart(TimeStamp *timeStamp){
#ifndef _WIN32
	gettimeofday(timeStamp, NULL);    
#else
	QueryPerformanceCounter(timeStamp);
#endif
}

double timeElapse(const TimeStamp timeStampStart, TimeStamp *timeStampEnd){
#ifndef _WIN32
	gettimeofday(timeStampEnd, NULL);    
	return (double)(timeStampEnd->tv_sec - timeStampStart.tv_sec)*1000.0 + (double)(timeStampEnd->tv_usec - timeStampStart.tv_usec)/1000.0;  //ms
#else
	QueryPerformanceCounter(timeStampEnd);
	return (double)(timeStampEnd->QuadPart - timeStampStart.QuadPart)/CPUFreq*1000.0;  //ms
#endif
}

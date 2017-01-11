#include "global.h"

CodingOption codingOption;

const char *parameters[PARAMETERNUM] = {
	"InputFile",			//original yuv
    "FramesToBeEncoded",
    "SequenceSize",			//QCIF: 176x144, CIF: 352x288
    "GOP",
    "Qindex",				//Quantization index (Range between 1 and 8)
    "WZBitstreamFile",		//output of WZ bitstream
	"SkipBlock",			//0: disable, 1: enable
	"IntraMode",			//0: disable, 1: enable
    "IntraQP",				//Quantization parameter for Intra block
	"DeblockingFilter",		//0: disable, 1: enable
	"KeyFrameSequence",		//reconstruct yuv by h264
	"IntraBitstreamFile"	//output of intra block bitstream
};

int FrameSize;
int Block44Num;
int Block88Num;
int Block44Height, Block44Width;
int Block88Height, Block88Width;
unsigned char **oriFrames;	//original sequence (only y)

EncodedFrame *encodedFrames;


int zigzag[BLOCKSIZE44] = {
	0, 1, 4, 8,
	5, 2, 3, 6,
	9,12,13,10,
	7,11,14,15,
};

//int SkipMin;
//int IntraMin;

//used for intra mode selection
Bitstream IntraBuffer;
int Recal_Intra = 1;
FILE *IntraFp;
unsigned char **reconsFrames;	//used to create SI
double *alpha_perframe;
double *Kmatrix;


double CPUFreq;

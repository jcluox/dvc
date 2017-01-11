#ifndef GLOBAL_H
#define GLOBAL_H

#include "defines.h"
#ifndef _WIN32
#include <sys/time.h>
#else
#include <windows.h>
#endif

typedef struct {
	char OriYuvFile[300];
	char WZBitStreamFile[300];
	char IntraBitStreamFile[300];
	char KeyFrameYuvFile[300];
	char KeyFrameInfo[300];
	char RDFile[300];
	char OutputDecodedYuvFile[300];
	int FramesToBeDecoded;
	int FrameWidth;
	int FrameHeight;
	int FrameRate;
	unsigned char GOP;
	unsigned char Qindex;
	unsigned char IntraQP;
	int ParallelMode;
	int UseCoreNum;
	char IntraModeExist;
	unsigned char DeblockingMode;

	int GPUid;
	int startFrame;
	int endFrame;
} CodingOption;

extern CodingOption codingOption;

extern const char *parameters[PARAMETERNUM];

extern int FrameSize;
extern int Block44Num;
extern int Block1616Num;	//macroblock number
extern int Block88Num;
extern int Block44Height, Block44Width;
extern int Block88Height, Block88Width;
extern int Block1616Height, Block1616Width;
extern unsigned char **oriFrames;  //original sequence (only y) used for PSNR calculation

typedef struct pix_pos{
    int  available;
    int  x;
    int  y;
    int  pos_x;
    int  pos_y;
} PixelPos;

typedef struct blockdata{
    int nonzero;// non zero number
    int mode;   // coding mode, intra or wz
    int bitcounter;
    int level[(BLOCKSIZE44+1)];
    int run[BLOCKSIZE44+1];
} BlockData;

typedef struct syntaxelement{
    int                 type;           //!< type of syntax element for data part.
    int                 value1;         //!< numerical value of syntax element
    int                 value2;         //!< for blocked symbols, e.g. run/level
    int                 len;            //!< length of code
    int                 inf;            //!< info part of UVLC code
    unsigned int        bitpattern;     //!< UVLC bitpattern
    int                 context;        //!< CABAC context
    int                 k;              //!< CABAC context for coeff_count,uv
} SyntaxElement;

//! Bitstream
typedef struct bitstream{
    // UVLC Decoding
    unsigned long frame_bitoffset;    //!< actual position in the codebuffer, bit-oriented, UVLC only
    unsigned long bitstream_length;   //!< over codebuffer lnegth, byte oriented, UVLC only
    //unsigned long bitmap_length;
    unsigned long prev_frame_bitoffset;
    // ErrorConcealment
    unsigned char *streamBuffer;      //!< actual codebuffer for read bytes
    int           ei_flag;            //!< error indication, 0: no error, else unspecified error
} Bitstream;

extern Bitstream IntraBS;

typedef struct {
	int ACRange[BLOCKSIZE44];		//use for dynamic AC coefficient range
	unsigned char *BlockModeFlag;	//0:WZ, 1:Intra, 2:Skip
	unsigned char *CRC;
	int *SyndromeByteLength;
	unsigned char **AccumulatedSyndrome;
	unsigned char skipMode;
	unsigned char intraMode;
	int skipBlockBits;
	int intraBlockBits;
	BlockData *blockInfo;
} EncodedFrame;

extern EncodedFrame *encodedFrames;

typedef struct {
	unsigned char *lowPassFrame;
	unsigned char *interpolateFrame;
	unsigned char hasReferenced;	//indicate this frame has been referenced or not
	unsigned char *reconsFrame;	//output sequence (only y)
	int bitRate;	//ac range + crc + syndrome
	double PSNR;
	double decodeTime;
	unsigned char isWZFrame;
} DecodedFrame;

extern DecodedFrame *decodedFrames;

typedef struct {
	double x, y;
} Point;

typedef struct {
	Point Past;
	Point Future;
	Point vector;
} MV;

/*typedef struct {
	float prob[2048];
} DISTR;*/

typedef struct {
	unsigned char *sideInfoFrame;
	int *residueFrame;
	int *reconsTransFrame;
	MV *mcroBlockMV;	//for each macroblock
	Point *Holes;	//for each macroblock
	MV *BlockMV;	//for each block
	double *MSE;
	float *alpha;	//for laplacian noise model
	//DISTR *noiseDist;	//correlation noise distribution
} SideInfo;

extern SideInfo sideInfo;

typedef struct  {
	float prob[ML_RANGE];
	int searchTop, searchLeft, searchRight, searchBottom;
	int searchWidth;
	int blkIdx;
	int probLen;
} SMF;

typedef struct {
	int	block44[BLOCKSIZE44];
} Trans;

typedef struct {
	int searchTop;
	int searchBottom;
	int searchLeft;
	int searchRight;
	int top, left;
	int searchWidth;
	int searchRange;
	int blkIdx; //會跳號  絕對位置
} SearchInfo;

extern SearchInfo *SearchInfo1616;

typedef struct {
	SearchInfo *searchInfo;
	int searchCandidate;
} SIR;

extern int zigzag[BLOCKSIZE44];

extern double CPUFreq;

#ifdef SHOW_TIME_INFO
extern double Total_overT_time;
extern double Total_ldpca_time;
extern double Total_CNM_time;
extern double Total_condBit_time;
extern double Total_updateSI_time;
extern double Total_sideCreate_time;
extern double Total_ML_time;
/*add by JSRF*/
typedef struct {
	int count;
	double toGPU_time;
	double Row_time;
	double Col_time;
	double BpCheck_time;
	double toCPU_time;
	double CPUloop_time;
	double checkResult_time;
	double InverseMatrix_time;
} LdpcInfo;
extern LdpcInfo *ldpcInfo;
extern int numCodes;
#endif

#ifndef _WIN32
typedef struct timeval TimeStamp;
#else
typedef LARGE_INTEGER TimeStamp;
#endif

void timeStart(TimeStamp *timeStamp);
double timeElapse(const TimeStamp timeStampStart, TimeStamp *timeStampEnd);

//extern double byte2bitTime;

#endif
#ifndef GLOBAL_H
#define GLOBAL_H

#include <stdio.h>
#include "defines.h"

typedef struct {
	char InputFile[300];
	char WZOutputFile[300];
	char KeyFrameYuvFile[300];
	char IntraOutputFile[300];
	int FramesToBeEncoded;
	int FrameWidth;
	int FrameHeight;
	unsigned char GOP;
	unsigned char Qindex;
	int EnableSkipBlock;
	int EnableIntraMode;
	unsigned char IntraQP;
	unsigned char DeblockingMode;
} CodingOption;
extern CodingOption codingOption;

extern const char *parameters[PARAMETERNUM];

extern int FrameSize;
extern int Block44Num;
extern int Block88Num;
extern int Block44Height, Block44Width;
extern int Block88Height, Block88Width;
extern unsigned char **oriFrames;	//original sequence (only y)


typedef struct pix_pos{
    int available;
    int pos_x;
    int pos_y;
} PixelPos;

typedef struct bit_counter
{
    int mb_total;
    unsigned short mb_mode;
    unsigned short mb_inter;
    unsigned short mb_cbp;
    unsigned short mb_delta_quant;
    int mb_y_coeff;
    int mb_uv_coeff;
    int mb_cb_coeff;
    int mb_cr_coeff;
    int mb_stuffing;
} BitCounter;

typedef struct blockdata{
    int nonzero;// non zero number
    int mode;   // coding mode, intra or wz
    BitCounter bits;
    int bitcounter;
    int level[(BLOCKSIZE44+1)];
    int run[BLOCKSIZE44+1];
} BlockData;

//Intra Bitstream
typedef struct bitstream{
    int     buffer_size;        //!< Buffer size      
    int     byte_pos;           //!< current position in bitstream;
    int     bits_to_go;         //!< current bitcounter
    unsigned char    byte_buf;           //!< current buffer for last written byte
    unsigned char    *streamBuffer;      //!< actual buffer for written bytes
} Bitstream;

typedef struct {
	int ACRange[BLOCKSIZE44];		//use for dynamic AC coefficient range
	unsigned char *CRC;
	unsigned char *skipBlockFlag;
	unsigned char *intraBlockFlag;
	int SyndromeByteLength;
	unsigned char **AccumulatedSyndrome;
	unsigned char isWZ;
	unsigned char skipMode;
	unsigned char intraMode;
	double *E;
    double *Z;
    int *Rspent;
    int intraCount;
    BlockData *blockInfo;
} EncodedFrame;
extern EncodedFrame *encodedFrames;


extern int zigzag[BLOCKSIZE44];
//extern int SkipMin;
//extern int IntraMin;

//used for intra mode selection
extern Bitstream IntraBuffer;
extern int Recal_Intra;
extern FILE *IntraFp;
extern unsigned char **reconsFrames;	//used to create SI
extern double *alpha_perframe;
extern double *Kmatrix;


extern double CPUFreq;

#endif
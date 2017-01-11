#ifndef LDPCA_KERNEL_H
#define LDPCA_KERNEL_H
#include <cuda.h>     //CUDA Driver API
#ifdef __cplusplus 
extern "C" { 
#endif

/* 
define LDPCA_SINGLE_BUFFER may cause race condition!!(according to CUDA Programming Guide 3.2 in [3.2.4.4])
Fortunately, it'll cause no harm to coding efficiency
On the contrary, message may ne update frome other thread in other cuda blocks, resulting codewords being converge more quickly
(the ordering of cuda block execution is scheduled by CUDA device)
Assume SPA algorithm is optimal & the soft input is correct, using message which is newer is usually better
*/
#define LDPCA_SINGLE_BUFFER   // have texture memory too

/*
slightly speed up only in CUDA capability > 1.3 , but usually have lower bitrate in both CIF & QCIF sequence
*/
#define LDPCA_HALF_FLOAT   

#define QCIF 1
#define CIF 2

/*
LDPCA_BLOCK_SIZE must be greater than hightest-rate-H's "row sum"(number of 1's in a row)
*/
#define LDPCA_BLOCK_SIZE 128	

/*
The header(early jump infomation) in device memory is resided in d_decodeInfo[0] to d_decodeInfo[LDPCA_DECODED_HEADER]
(Utilize coalesced memory access)
*/
#define LDPCA_DECODED_HEADER 2	 

#define JUMP_INFO 0x02           //header info
#define FAILED_INFO 0x01         //header info
#define INFO_SWAP 1              //swap current header index resided in d_decodedInfo between 0 & 1

typedef struct __align__(4) {
	short syndIdx;
	short softIdx;
} Index2;

typedef struct __align__(4) {
	short colIdx;
	unsigned char syndrome;
	unsigned char rowDeg;
} RowInfo;

typedef struct __align__(4) {
	short next1;
	short next2;
} ColInfo;

#ifdef LDPCA_HALF_FLOAT
	typedef unsigned short Msg;	
#else
	typedef float Msg;		
#endif

void init_kernel_RT(CUstream stream, int nCode, int Size, int numBlock, Index2*   HinitIdx, char*  syndromeByte, float*  softInput, RowInfo*  Hinfo,  float*  softInputPadding, Msg*  InputMessage);
void update_kernel_RT(CUstream stream, int nCode, int Size, int numBlock, Msg*  InputMessage, float*  softInputPadding, ColInfo*  Vinfo, RowInfo*  Hinfo);
void check_kernel_RT(CUstream stream, int nCode, int Size, int numBlock, int decodedHeaderIdx, const float*  softInputPadding, ColInfo*  Vinfo, RowInfo*  Hinfo, unsigned char*  decodedInfo);
void check_update_kernel_RT(CUstream stream, int nCode, int Size, int numBlock, int decodedHeaderIdx, Msg*  InputMessage, float*  softInputPadding, ColInfo*  Vinfo, RowInfo*  Hinfo, unsigned char*  decodedInfo);

#ifdef __cplusplus 
}
#endif

#endif
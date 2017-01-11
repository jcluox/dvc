//JSRF: 更改inv_H_matrix型態，以及他的矩陣乘法演算法

#ifndef READ_LDPCA_LADDER_FILE_LDPCA_DECODER_H
#define READ_LDPCA_LADDER_FILE_LDPCA_DECODER_H

#define BLOCK_SIZE_CFG 128	//(greater than 99) 

//ladder file parameter for ldpca decoder
extern int numCodes, codeLength, numEdges, period;
extern int* jc;
extern int* bitsPerPeriod;
extern int** idxPerPeriod;
extern int** ir;
extern int* syndLength;
extern float* codeRate;
//inverse H matrix for ldpca decoder
extern unsigned char* inv_H_matrix;


//CUDA ladder data(CPU variables)
extern int blockSize, MAX_BlockNumber;
//for each H matrix
extern int *numberOfBlocks;  //v
extern int **rmnIndex0, **rmnIndex1, **rmnIndex2;
extern int **pnext;
extern int **prowCount;
extern int **pcolnext0, **pcolnext1;
extern int **rsum;
extern int **synhead;
extern int **decodedIndex;

//void CUDA_Memory_Alloc();
void ReadLadder_decoder_CUDA(char* LadderFileNameCUDA);
void ReadLadder_decoder(char* LadderFileName, char* InverseHmatrixFileName);
void freeLdpcaParam();
void freeLdpcaParam_CUDA();
#endif
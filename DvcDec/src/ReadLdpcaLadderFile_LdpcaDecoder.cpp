//JSRF: 更改inv_H_matrix型態，以及他的矩陣乘法演算法
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "ReadLdpcaLadderFile_LdpcaDecoder.h"
#include "LdpcaDecoder.h"  //debug
#include "global.h"


//ladder file parameter for ldpca decoder
//                  N                   length of transmission pattern
int numCodes, codeLength, numEdges, period;
int* jc;				//cumulative sum of column sum of H[numEdges+1]
int* bitsPerPeriod;		//number of bits sent per period
int** idxPerPeriod;		//INDEX of bits sent in each period [bitsPerPeriod] 過去已經送過的index，這裡也會重複再寫一次，一直累加上去
int** ir;			//row indices of ones in H [numEdges]
int* syndLength;
float* codeRate;
//inverse H matrix for ldpca decoder
unsigned char* inv_H_matrix;

//CUDA ladder data(CPU variables)
int blockSize, MAX_BlockNumber;
//for each H matrix
int *numberOfBlocks;
int **rmnIndex0, **rmnIndex1, **rmnIndex2;
int **pnext;
int **prowCount;
int **pcolnext0, **pcolnext1;
int **rsum;
int **synhead;
int **decodedIndex;

//小小白
void ReadLadder_decoder_CUDA(char* LadderFileNameCUDA)
{
	int i, k, ActualSize, N, m;
	
	FILE* pLadderFile = fopen(LadderFileNameCUDA, "r");
	if(pLadderFile == NULL){	
		printf("CUDA File %s doesn't exist\n", LadderFileNameCUDA);
	#ifdef _WIN32
		system("pause");
	#endif
		exit(1);
	}

	N = codeLength;

	fscanf(pLadderFile, "%d", &blockSize);  //128
	fscanf(pLadderFile, "%d", &MAX_BlockNumber);  //48
	//the * indicates that the field is to be read but ignored
	fscanf(pLadderFile, "%*d");	//numCodes
    fscanf(pLadderFile, "%*d");	//codeLength
    fscanf(pLadderFile, "%*d");	//numEdges
    fscanf(pLadderFile, "%*d");	//period	

	numberOfBlocks = (int *)malloc(sizeof(int) * numCodes);

	rmnIndex0 = (int**)malloc(sizeof(int*) * numCodes);
	rmnIndex1 = (int**)malloc(sizeof(int*) * numCodes);
	rmnIndex2 = (int**)malloc(sizeof(int*) * numCodes);
	decodedIndex = (int**)malloc(sizeof(int*) * numCodes);
	for(i=0; i<numCodes; i++)
	{
		rmnIndex0[i] = (int*)malloc(sizeof(int) * N);
		rmnIndex1[i] = (int*)malloc(sizeof(int) * N);
		rmnIndex2[i] = (int*)malloc(sizeof(int) * N);	
		decodedIndex[i] = (int*)malloc(sizeof(int) * N);
	}
	
	pnext = (int**)malloc(sizeof(int*) * numCodes);
	prowCount = (int**)malloc(sizeof(int*) * numCodes);
	pcolnext0 = (int**)malloc(sizeof(int*) * numCodes);
	pcolnext1 = (int**)malloc(sizeof(int*) * numCodes);
	rsum = (int**)malloc(sizeof(int*) * numCodes);
	synhead = (int**)malloc(sizeof(int*) * numCodes);	

	//for each H matrix
	for(i=0; i<numCodes; i++)
	{
		m = syndLength[i];		
	
		fscanf(pLadderFile, "%d", numberOfBlocks+i);  //48 40 39 ...
		ActualSize = numberOfBlocks[i] * blockSize;	  //48 * 128 >= edgeNum
	
		pnext[i] = (int*)malloc(sizeof(int) * ActualSize);
		prowCount[i] = (int*)malloc(sizeof(int) * ActualSize);
		pcolnext0[i] = (int*)malloc(sizeof(int) * ActualSize);
		pcolnext1[i] = (int*)malloc(sizeof(int) * ActualSize);
		rsum[i] = (int*)malloc(sizeof(int) * m); //要把cuda if拿掉，這裡的array長度要+  (blockSize-1)
		synhead[i] = (int*)malloc(sizeof(int) * m); //要把cuda if拿掉，這裡的array長度要+  (blockSize-1)
	
		//rmnIndex
		for(k = 0; k < N; k++)
			fscanf(pLadderFile, "%d", rmnIndex0[i]+k);  //0<= index < ActualSize
		for(k = 0; k < N; k++)
			fscanf(pLadderFile, "%d", rmnIndex1[i]+k);  //0<= index < ActualSize
		for(k = 0; k < N; k++)
			fscanf(pLadderFile, "%d", rmnIndex2[i]+k);  //0<= index < ActualSize

		//pnext, prowCount, pcolnext0, pcolnext1
		for(k = 0; k < ActualSize; k++)  //(1 2 3... 98 0 -1 -1 -1 -1) * numberOfBlocks[i]
			fscanf(pLadderFile, "%d", pnext[i]+k);
		
		//(每個H的row的一的總數 – 1,每個H的row的一的總數 – 1,每個H的row的一的總數 – 1...)  * numberOfBlocks[i]  
		//其中(每個H的row的一的總數-1) x prowCount次
		//重複寫這麼多次是因為 kernal以edge為單位去做threading，為了讓每個thread都透過自己的id知道自己的row有幾個1
		for(k = 0; k < ActualSize; k++) 
			fscanf(pLadderFile, "%d", prowCount[i]+k);
		for(k = 0; k < ActualSize; k++)
			fscanf(pLadderFile, "%d", pcolnext0[i]+k); //0<= index < ActualSize
		for(k = 0; k < ActualSize; k++)
			fscanf(pLadderFile, "%d", pcolnext1[i]+k); //0<= index < ActualSize
	
		//rsum, synhead, decodedIndex
		for(k = 0; k < m; k++)
			fscanf(pLadderFile, "%d", rsum[i]+k);
		for(k = 0; k < m; k++)
			fscanf(pLadderFile, "%d", synhead[i]+k);
		for(k = 0; k < N; k++)
			fscanf(pLadderFile, "%d", decodedIndex[i]+k);
	}

	fclose(pLadderFile);	
}

//#define WORDLEN(x) (((x)&3)==0? (x) : ((x) + 4 - ((x)&3)) )
#define BIT_LEN2WORD_LEN(x) ( ((x)&31)==0? ( (x)>>5 ) : ( ( (x)>>5 ) + 1) )

void ReadLadder_decoder(char* LadderFileName, char* InverseHmatrixFileName){
	int k, j, nCode;
	int *pData;	 
//	unsigned char *uData;

	//////////////////////////////////////////////////
	//--------------read ladder file----------------//
	//////////////////////////////////////////////////
	FILE* pLadderFile = fopen(LadderFileName, "r");
	if(pLadderFile == NULL){
//		sprintf(errorText, "File %s doesn't exist", LadderFileName);	
//		errorMsg(errorText);
	}
	FILE* pInverseFile = fopen(InverseHmatrixFileName, "rb");
	if(pInverseFile == NULL){
//		sprintf(errorText, "File %s doesn't exist", InverseHmatrixFileName);	
//		errorMsg(errorText);
	}
	
	fscanf(pLadderFile, "%d", &numCodes);
    fscanf(pLadderFile, "%d", &codeLength);
    fscanf(pLadderFile, "%d", &numEdges);
    fscanf(pLadderFile, "%d", &period);
      
	jc = (int *)malloc(sizeof(int) * (codeLength+1));

	for (k = 0; k < codeLength+1; k++)
        fscanf(pLadderFile, "%d", jc+k); 

	bitsPerPeriod = (int *)malloc(sizeof(int) * numCodes);

	idxPerPeriod = (int **)malloc(numCodes*sizeof(int *)+period*numCodes*sizeof(int));
    for(j=0, pData = (int *)(idxPerPeriod+numCodes); j < numCodes ; j++, pData += period)
        idxPerPeriod[j] = pData;

	ir = (int **)malloc(numCodes*sizeof(int *)+numEdges*numCodes*sizeof(int));
    for(j=0, pData = (int *)(ir+numCodes); j < numCodes ; j++, pData += numEdges)
        ir[j] = pData;

	
	syndLength = (int *)malloc(sizeof(int) * numCodes);
	codeRate = (float *)malloc(sizeof(float) * numCodes);


	for (nCode = 0; nCode < numCodes; nCode++)
	{
		fscanf(pLadderFile, "%d", bitsPerPeriod+nCode);		
		for (k = 0; k < bitsPerPeriod[nCode]; k++)
			fscanf(pLadderFile, "%d", idxPerPeriod[nCode]+k);
		for (k = 0; k < numEdges; k++)
			fscanf(pLadderFile, "%d", ir[nCode]+k);
		syndLength[nCode] = (codeLength/period)*bitsPerPeriod[nCode];
		codeRate[nCode] = ((float) syndLength[nCode])/((float) codeLength);
	}
	fclose(pLadderFile);
	
	///////////////////////////////////////////////////////
	//--------------read inverse H matrix----------------//
	///////////////////////////////////////////////////////
	
	//allocate memory for inverse H matrix  (SSW version for .bat)
/*
	inv_H_matrix = (unsigned char *)malloc(codeLength*codeLength*sizeof(unsigned char));

	for(k=0;k<codeLength*codeLength;k++)
		fscanf(pInverseFile, "%d", &inv_H_matrix[k]);
*/	
	
	inv_H_matrix = (unsigned char*)malloc(sizeof(int) * BIT_LEN2WORD_LEN(codeLength) * codeLength);
	rewind(pInverseFile);
	size_t total = fread(inv_H_matrix, sizeof(int), BIT_LEN2WORD_LEN(codeLength) * codeLength , pInverseFile);	




	//inv_H_matrix = (unsigned char*)malloc(sizeof(unsigned char) * BIT_LEN2BYTE_LEN(codeLength) * codeLength);
	//fread(inv_H_matrix, sizeof(unsigned char), BIT_LEN2BYTE_LEN(codeLength) * codeLength, pInverseFile);
	//printf("%x %x %x %x \n",inv_H_matrix[0],inv_H_matrix[1],inv_H_matrix[2],inv_H_matrix[3]);
	//printf("%x \n",((int*)inv_H_matrix)[0]);getchar();

	//=============================製作Inverse_Matrix_H_Reg6336.bin   Inverse_Matrix_H_Reg1584.bin    bit版本======================
	//bin generation	
	//unsigned char *inv_H = (unsigned char *)malloc(codeLength*codeLength*sizeof(unsigned char));

	//for(k=0; k<codeLength*codeLength; k++)
	//	fscanf(pInverseFile, "%d ", &inv_H[k]);	

	//fclose(pInverseFile);
	//pInverseFile = fopen("LDPCA_LadderFile/Inverse_Matrix_H_Reg6336.bin","w");
	//int size=0;
	//for(int l=0; l<codeLength; l++){
	//	int numBytes;
	//	unsigned char *inv_H_byte = BitToByte(inv_H + l*codeLength, codeLength, &numBytes);

	//	fwrite(inv_H_byte, sizeof(char), numBytes, pInverseFile);

	//	int padding = WORDLEN(numBytes) - numBytes;
	//	char zero[4]={0,0,0,0};
	//	fwrite(zero, sizeof(char), padding, pInverseFile );
	//	//printf("===%d + %d===", numBytes, padding);getchar();
	//	free(inv_H_byte);
	//	size+=padding+numBytes;
	//}printf("file size:%x bytes\n",size);

	//free(inv_H);
	//fclose(pInverseFile);
	//pInverseFile = fopen("LDPCA_LadderFile/Inverse_Matrix_H_Reg1584.bin","r");

	//fseek (pInverseFile, 0, SEEK_END );
	//size = ftell(pInverseFile);
	//fseek (pInverseFile, 0, SEEK_SET );
	//size = size - ftell(pInverseFile);

	//printf("file size:%x bytes\n",size);
	//fclose(pInverseFile);

	//exit(1);	
	//*/
	fclose(pInverseFile);
}

void freeLdpcaParam(){

	free(jc);
	free(bitsPerPeriod);
	free(idxPerPeriod);
	free(ir);
	free(syndLength);
	free(codeRate);
	free(inv_H_matrix);
}

void freeLdpcaParam_CUDA(){

	free(numberOfBlocks);	

	for(int j=0; j<numCodes; j++)
	{
		free(pnext[j]);
		free(prowCount[j]);
		free(pcolnext0[j]);
		free(pcolnext1[j]);
		free(rsum[j]);
		free(synhead[j]);
		free(decodedIndex[j]);
		free(rmnIndex0[j]);
		free(rmnIndex1[j]);
		free(rmnIndex2[j]);
	}
	free(rmnIndex0);
	free(rmnIndex1);
	free(rmnIndex2);
	free(pnext);
	free(prowCount);
	free(pcolnext0);
	free(pcolnext1);
	free(rsum);
	free(synhead);

	free(decodedIndex);
}

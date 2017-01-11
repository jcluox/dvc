#include <string.h>
#include "ReadLdpcaLadderFile_LdpcaDecoder.h"
#include "LdpcaDecoder.h"
#include "LDPCA_cuda.h"
#include "LDPCA_kernel.h"
#include "global.h"
#include "error.h"
#include "cutil_inline_drvapi.h"

#define ALIGN_128(size) ((((size)&127)==0)? (size): ((size) + 128-((size)&127)))  //optimized for fermi
//#define ALIGN_128(size) (size)
//#define RUNTIME_API
#ifdef RUNTIME_API
	void bind_txture_RT(Msg* message, int size);//test
#endif

int LDPC_iterations=0;
//#define MAX_ITERATIONS 100

typedef struct {
	CUfunction init;
	CUfunction update;
	CUfunction check_update;
	CUfunction check;
} CUfuncs;

CUdevice hDevice;
CUcontext hContext;
CUmodule hModule;
CUfuncs *BPalgo;
CUtexref hTexRef;

int num_updates_per_memcpy_overlap;
int cuda_concurrent_copy_execution;

bool InitCUDA()
{
	cudaDeviceProp deviceProp;
	cuInit(0);  //driver API
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);	
	//printf("deviceCount=%d\n",deviceCount);
    if(deviceCount == 0) {
        fprintf(stderr, "  There is no device.\n");
        return false;
    }
    int deviceID = codingOption.GPUid;
    CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, deviceID));
	if (deviceProp.major == 9999 && deviceProp.minor == 9999){
        printf("  There is no device supporting CUDA.\n");
		return false;
	}
    else if (deviceCount == 1)
        printf("  There is 1 device supporting CUDA\n");
    else
        printf("  There are %d devices supporting CUDA\n", deviceCount);
//		printf("  CUDA concurrent Kernel execution: %d\n", deviceProp.concurrentKernels);
	printf("  Using device %d (%s) with ",deviceID, deviceProp.name);
	printf("CUDA Capability : %d.%d\n", deviceProp.major, deviceProp.minor);			
	if(deviceProp.deviceOverlap){
		if(codingOption.FrameWidth == 176)	
			num_updates_per_memcpy_overlap = 4;  //optimaized for C2050 QCIF sequence
		else
			num_updates_per_memcpy_overlap = 3;  //optimaized for C2050 CIF sequence
		cuda_concurrent_copy_execution = 1;
	}
	else{
		num_updates_per_memcpy_overlap = -1;
		cuda_concurrent_copy_execution = 0;
	}
    cudaSetDevice(deviceID);	
	
	// create CUDA device & context
	cutilDrvSafeCall(cuDeviceGet(&hDevice, deviceID));
	cutilDrvSafeCall(cuCtxCreate(&hContext, 0, hDevice));
	#ifndef _WIN32 		
		#ifdef FERMI
			printf("LDPCA_kernel_sm20.cubin\n");
			cutilDrvSafeCall(cuModuleLoad(&hModule, "LDPCA_kernel_sm20.cubin"));	
		#else
			printf("LDPCA_kernel_sm10.cubin\n");
			cutilDrvSafeCall(cuModuleLoad(&hModule, "LDPCA_kernel_sm10.cubin"));	
		#endif
	#else
		//cutilDrvSafeCall(cuModuleLoad(&hModule, "LDPCA_kernel.ptx"));
		cutilDrvSafeCall(cuModuleLoad(&hModule, "LDPCA_kernel.cubin"));
	#endif
    return true;
}


int *numBlocksH;					//[numCodes]
int **syndPos;                        // (varies across codes) syndrome position reodering index => position 'k' is moved to position 'syndPos[k]'
unsigned char *syndrome_reoder_byte;
unsigned char *decodedInfo_pinned;      //[0] 給所有checkMessage中的thread使用(記錄自己有沒有positive early jump)
							            //[1] 給所有checkMessage中的thread使用(記錄自己有沒有negative early jump)

CUstream updateStream;
CUstream checkStream;

//CUDA ladder data(GPU variables)
RowInfo **d_Hinfo;      //(varies across codes) Horizontal processing structure
Index2 **d_HinitIdx;      //(varies across codes) soft input initialization for Horizontal processing
ColInfo **d_Vinfo;      //(varies across codes) Vertical processing structure
char   *d_syndromePadding;
float  *d_softInputPadding;      //(varies across codes) Vertical processing structure
float *d_LR_intrinsic;     //soft input, from coiseModel_kernel.cu
char   *d_syndromeByte;    //(N/8)

Msg *d_Hmessage;          
Msg *d_Vmessage;          
unsigned char *d_decodedInfo;      //[0] 給所有checkMessage中的thread使用(記錄自己有沒有positive early jump)
							       //[1] 給所有checkMessage中的thread使用(記錄自己有沒有negative early jump)
char  *d_VmessageBit;      //bit   node to check node message  (Vtid) final Vertical  processing output this variable (decoded bit)  (no use)


typedef struct {
	int N,M,nCode;
	int *sumCol;					//[N]
	int *accumSumCol;				//[N+1]  jc;
	int **idxInCol;					//[N][sumCol[N]];  ir[c][ jc[i] ]
	int *_idxInCol;					//[numEdges]
	int *sumRow;					//[M]
	int *accumSumRow;				//[M+1]
	int **idxInRow;					//[M][sumRow[M]]
	int *_idxInRow;                 //[numEdges]
} MtxStruct;

typedef struct {
	int preIdx;
	int sumRow;
} SortPack;
/*
1. 做syndrome reodering
2. 把每個byte中的第一個bits取出來，濃縮再一個*char的bitstream裡面
*/
unsigned char*  syndromeReorder(const int *syndPos, const unsigned char *Bits, const int BitLength, int *numBytes){
	int i, remainder;
	int ByteLength;
	unsigned char *Bytes = syndrome_reoder_byte;
	
	ByteLength = BitLength >> 3; //BitLength / 8
	remainder = BitLength & 7;	//BitLength % 8
	if(remainder != 0)
		ByteLength++;

	//Bytes = (unsigned char *)malloc(sizeof(unsigned char) * ByteLength);	
	//cudaHostAlloc ( &Bytes, sizeof(unsigned char) * ByteLength, cudaHostAllocDefault); //cudaHostAllocWriteCombined
	*numBytes = ByteLength;	//output number of bytes

	int byteIdx = 0;
	unsigned char tmp = 0;
	for(i=0; i<BitLength; i++){
		tmp |= Bits[syndPos[i]];
		if(((i+1)&7) == 0){	//(i+1)%8 == 0
			Bytes[byteIdx++] = tmp;
			tmp = 0;
		}
		else
			tmp <<= 1;
	}
	if(remainder != 0)
		Bytes[byteIdx++] = tmp<<(7-remainder);

	return Bytes;
}
int posComp(const void *v1, const void *v2){
	const int a = *(int *)v1;
	const int b = *(int *)v2;
	if(a<b)
		return -1;
	else if(a>b)
		return 1;
	else
		return 0;
}


int getRowPos(int r, int c, int *idxInRow, int len){
	return (int*)bsearch(&c, idxInRow, len, sizeof(int), posComp) - idxInRow;
}

//input x>=1
//output y,  y>=x & y=2^k
int get2pow(int x, int *pow){
	int y = 1;
	if(pow!=NULL)
		*pow = 0;
	while(y<x){
		y = (y<<1);
		if(pow!=NULL)
			(*pow)++;
	}
	return y;
}
int getBits(int x){
	int y = 1;
	int pow = 0;
	while(y<x){
		y = (y<<1);
		pow++;
	}
	return pow;
}
int get_Hinfo_size(MtxStruct *parityH){
	int M = parityH->M;
	int *sumRow = parityH->sumRow;
	int tx=0;          //thread index within a block
	int numBlocksH = 1;  //block number	
	int preActiveNum = 0;
	for(int k = 0; k < M; k++){  
		int allNum;
		int activeNum = sumRow[k];
		allNum = get2pow(sumRow[k], NULL);

		//假設每一條sumRow都不會超過LDPCA_BLOCK_SIZE (設定的CUDA block size)
		if( allNum > LDPCA_BLOCK_SIZE ){
			char str[512];
			sprintf(str,"LDPCA_BLOCK_SIZE=%d(threads number in CUDA blocks) is smaller then %d\n", LDPCA_BLOCK_SIZE, allNum);
			errorMsg(str);
		}
		if(tx + allNum -1  >  LDPCA_BLOCK_SIZE - 1  || activeNum < preActiveNum ){   // if this block does not have sufficient space
			//change to next block in CUDA										// or this is a new row having different rowSum
			tx = 0;
			numBlocksH++;
		}
		tx  += allNum;
		preActiveNum = activeNum;
	}
	return numBlocksH*LDPCA_BLOCK_SIZE;
}

MtxStruct *getHstruct(int nCode){
	MtxStruct *parityH = (MtxStruct*)malloc(sizeof(MtxStruct)); 
	parityH->nCode = nCode;
	int N,M;
	N = parityH->N = codeLength;
	M = parityH->M = syndLength[nCode];
	
	//get sum in each Colume
	parityH->sumCol = (int*)malloc( sizeof(int)*N );
	parityH->accumSumCol = (int *)malloc(sizeof(int)*(N+1));
	parityH->_idxInCol = (int*)malloc( sizeof(int)*numEdges );
	parityH->idxInCol = (int **)malloc(sizeof(int*)*N);	
	memcpy(parityH->accumSumCol, jc, sizeof(int)*(N+1));
	for(int i=0; i<N; i++)
		parityH->sumCol[i] = parityH->accumSumCol[i+1] - parityH->accumSumCol[i];
//for(int i=0; i<N; i++){
//	printf("%d ", sumCol[i]);
//}getchar();
	parityH->sumRow = (int*)malloc( sizeof(int)*M );
	parityH->accumSumRow = (int*)malloc( sizeof(int)*(M+1) );
	parityH->_idxInRow = (int*)malloc( sizeof(int)*numEdges );
	parityH->idxInRow = (int**)malloc( sizeof(int*)*M );
	//get Colume index in each Colume having 1		
	//printf("nCode= %d\n",nCode);
	memcpy(parityH->_idxInCol, ir[nCode], sizeof(int)*numEdges);
	for(int i=0; i<N; i++){
		parityH->idxInCol[i] = &parityH->_idxInCol[ jc[i] ];	
	}
	//get sum in each Row
	memset(parityH->accumSumRow, 0, sizeof(int)*(M+1));
	for(int i=0; i<numEdges; i++){
		parityH->accumSumRow[ir[nCode][i]]++;
	} 
	//accumulate sum Row
	parityH->sumRow[0] = parityH->accumSumRow[0];
	parityH->accumSumRow[0] = 0;		
	for(int i=1; i<M; i++){
		parityH->sumRow[i] = parityH->accumSumRow[i];
		parityH->accumSumRow[i] = parityH->accumSumRow[i-1] + parityH->sumRow[i-1];
	}
	parityH->accumSumRow[M] = parityH->accumSumRow[M-1] + parityH->sumRow[M-1];
//for(int i=0; i<M; i++){
//	printf("%d ",parityH->sumRow[i]);
//}getchar();

	//get Row index in each Row having 1
	int count=0;
	for(int y=0; y<N; y++){					//水平方向index
		for(int l=0; l<parityH->sumCol[y]; l++){
			int x = ir[nCode][count++];			//縱方向index
			parityH->_idxInRow[parityH->accumSumRow[x]++] = y;
		}
	}
	for(int m=0; m < M; m++){
		parityH->accumSumRow[m] -= parityH->sumRow[m];	
		parityH->idxInRow[m] = &(parityH->_idxInRow[parityH->accumSumRow[m]]);
	}
	return parityH;
}

void freeHstruct(MtxStruct *parityH){
	free(parityH->sumCol);
	free(parityH->accumSumCol);
	free(parityH->idxInCol);
	free(parityH->_idxInCol);		

	free(parityH->sumRow);
	free(parityH->accumSumRow);
	free(parityH->_idxInRow);
	free(parityH->idxInRow);
	free(parityH);
}
int rowComp(const void *v1, const void *v2){
	const SortPack a = *(SortPack *)v1;
	const SortPack b = *(SortPack *)v2;
	if(a.sumRow < b.sumRow)
		return 1;
	else if(a.sumRow > b.sumRow)
		return -1;
	else
		return 0;
}
int idxComp (const void * a, const void * b){
  return ( *(int*)a - *(int*)b );
}

/* row sum由大至小排列 */
void HstructReorder(MtxStruct *parityH, int *IdxMapping){
	//assert(IdxMapping!=NULL);
	////test
	//for(int i=0; i<parityH->M; i++)
	//	IdxMapping[i] = i;
	//return;

	int M = parityH->M;
	int N = parityH->N;
	SortPack *rowSumArray = (SortPack*) malloc(sizeof(SortPack)*M);
	int *inverseIdx = (int*)malloc( sizeof(int)*M );
	int **oldIdxInRow = (int**)malloc( sizeof(int*)*M );
	memcpy(oldIdxInRow, parityH->idxInRow, sizeof(int*)*M);

	for(int i=0; i<M; i++){
		rowSumArray[i].preIdx = i;
		rowSumArray[i].sumRow = parityH->sumRow[i];
	}
	qsort(rowSumArray, M, sizeof(SortPack), rowComp);
	//for(int i=0; i<M; i++){
	//	printf("[%d]=%d %d\n",rowSumArray[i].preIdx , rowSumArray[i].sumRow, parityH->sumRow[i]);
	//}getchar();	
	
	/* Row structure reordering */	
	parityH->accumSumRow[0] = 0;
	for(int newIdx=0; newIdx<M; newIdx++){
		//revise sumRow structure
		parityH->sumRow[newIdx] = rowSumArray[newIdx].sumRow;
		parityH->accumSumRow[newIdx+1] = parityH->accumSumRow[newIdx] + parityH->sumRow[newIdx];


		//revise idxInRow structure
		int preIdx = rowSumArray[newIdx].preIdx;
		parityH->idxInRow[newIdx] =  oldIdxInRow[preIdx];

		//create inverse index
		inverseIdx[preIdx] = newIdx;
		IdxMapping[newIdx] = preIdx;

		////debug
		//for(int i=0; i<parityH->sumRow[newIdx]-1; i++){
		//	if(parityH->idxInRow[newIdx][i] > parityH->idxInRow[newIdx][i+1]){
		//		printf("%d<->%d\n",preIdx,newIdx);
		//		printf("%d %d\n",parityH->idxInRow[newIdx][i],parityH->idxInRow[newIdx][i+1]);
		//		getchar();
		//	}
		//}
	}

	/* Col structure reordering - swap every edge(1) position */	
	for(int i=0; i<N; i++){  //for each column
		for(int j=0; j<parityH->sumCol[i]; j++){  //for each row in this colume
			int preIdx = parityH->idxInCol[i][j];			
			parityH->idxInCol[i][j] = inverseIdx[preIdx];  //revise previous index			
		}
		qsort(parityH->idxInCol[i], parityH->sumCol[i], sizeof(int), idxComp);
		////debug
		//for(int j=0; j<parityH->sumCol[i]-1; j++){
		//	assert(parityH->idxInCol[i][j] <= parityH->idxInCol[i][j+1]);
		//}
	}

	free(inverseIdx);
	free(oldIdxInRow);
	free(rowSumArray);
	return;
}





void load_ldpc_funcs(CUfuncs *BPalgo, int nCode, int Size){
	char functionName[256];
	//init_kernel_RT(stream, 0, QCIF, numBlocksH[nCode], d_HinitIdx[nCode], d_syndromeByte, d_softInput, d_Hinfo[nCode], d_softInputPadding, d_Hmessage);
	sprintf(functionName, "_Z18InitMessage_kernalILi%dELi%dEEvP6Index2PcPfP7RowInfoS3_Pt", nCode, Size);	
						  
	cutilDrvSafeCall(cuModuleGetFunction(&BPalgo->init, hModule, functionName));
	cuFuncSetBlockShape(BPalgo->init, LDPCA_BLOCK_SIZE, 1, 1);//numBlocksH[nCode], LDPCA_BLOCK_SIZE
	// set up parameter values
    int offset = 0;
    cutilDrvSafeCallNoSync(cuParamSetv(BPalgo->init, offset, &d_HinitIdx[nCode], sizeof(void*)));	
    offset += sizeof(void*);

	offset = (offset + __alignof(void*) - 1) & ~(__alignof(void*) - 1); // adjust offset to meet alignment requirement
    cutilDrvSafeCallNoSync(cuParamSetv(BPalgo->init, offset, &d_syndromeByte, sizeof(void*)));	
    offset += sizeof(void*);

	offset = (offset + __alignof(void*) - 1) & ~(__alignof(void*) - 1); // adjust offset to meet alignment requirement
    cutilDrvSafeCallNoSync(cuParamSetv(BPalgo->init, offset, &d_LR_intrinsic, sizeof(void*)));	
	//cutilDrvSafeCallNoSync(cuParamSetv(BPalgo->init, offset, &d_softInput, sizeof(void*)));	
    offset += sizeof(void*);

	offset = (offset + __alignof(void*) - 1) & ~(__alignof(void*) - 1); // adjust offset to meet alignment requirement
    cutilDrvSafeCallNoSync(cuParamSetv(BPalgo->init, offset, &d_Hinfo[nCode], sizeof(void*)));	
    offset += sizeof(void*);

	offset = (offset + __alignof(void*) - 1) & ~(__alignof(void*) - 1); // adjust offset to meet alignment requirement
    cutilDrvSafeCallNoSync(cuParamSetv(BPalgo->init, offset, &d_softInputPadding, sizeof(void*)));	
    offset += sizeof(void*);

	offset = (offset + __alignof(void*) - 1) & ~(__alignof(void*) - 1); // adjust offset to meet alignment requirement
    cutilDrvSafeCallNoSync(cuParamSetv(BPalgo->init, offset, &d_Hmessage, sizeof(void*)));	
    offset += sizeof(void*);

	cuParamSetSize(BPalgo->init, offset);

	//void update_kernel_RT(stream, nCode, QCIF, numBlocksH[nCode], d_Hmessage, d_softInputPadding, d_Vinfo[nCode], d_Hinfo[nCode]);
	#ifdef LDPCA_SINGLE_BUFFER
		sprintf(functionName, "_Z20UpdateMessage_kernalILi%dELi%dEEvPtPfP7ColInfoP7RowInfo", nCode, Size);							  		
	#else
		sprintf(functionName, "_Z20UpdateMessage_kernalILi%dELi%dEEvPtS0_PfP7ColInfoP7RowInfo", nCode, Size);    							  
	#endif
	cutilDrvSafeCall(cuModuleGetFunction(&BPalgo->update, hModule, functionName));
	cuFuncSetBlockShape(BPalgo->update, LDPCA_BLOCK_SIZE, 1, 1);//numBlocksH[nCode], LDPCA_BLOCK_SIZE
	// set up parameter values
    offset = 0;

	#ifndef LDPCA_SINGLE_BUFFER		
		cutilDrvSafeCallNoSync(cuParamSetv(BPalgo->update, offset, &d_Vmessage, sizeof(void*)));	
		offset += sizeof(void*);
		offset = (offset + __alignof(void*) - 1) & ~(__alignof(void*) - 1); // adjust offset to meet alignment requirement
	#endif

    cutilDrvSafeCallNoSync(cuParamSetv(BPalgo->update, offset, &d_Hmessage, sizeof(void*)));	
    offset += sizeof(void*);

	offset = (offset + __alignof(void*) - 1) & ~(__alignof(void*) - 1); // adjust offset to meet alignment requirement
    cutilDrvSafeCallNoSync(cuParamSetv(BPalgo->update, offset, &d_softInputPadding, sizeof(void*)));	
    offset += sizeof(void*);

	offset = (offset + __alignof(void*) - 1) & ~(__alignof(void*) - 1); // adjust offset to meet alignment requirement
    cutilDrvSafeCallNoSync(cuParamSetv(BPalgo->update, offset, &d_Vinfo[nCode], sizeof(void*)));	
    offset += sizeof(void*);

	offset = (offset + __alignof(void*) - 1) & ~(__alignof(void*) - 1); // adjust offset to meet alignment requirement
    cutilDrvSafeCallNoSync(cuParamSetv(BPalgo->update, offset, &d_Hinfo[nCode], sizeof(void*)));	
    offset += sizeof(void*);

	cuParamSetSize(BPalgo->update, offset);
	
	//void check_update_kernel_RT(stream, nCode, QCIF, numBlocksH[nCode], 0, d_Hmessage, d_softInputPadding, d_Vinfo[nCode], d_Hinfo[nCode], d_decodedInfo);	
	#ifdef LDPCA_SINGLE_BUFFER
		sprintf(functionName, "_Z25CheckUpdateMessage_kernalILi%dELi%dEEviPtPfP7ColInfoP7RowInfoPh", nCode, Size);
	#else
		sprintf(functionName, "_Z25CheckUpdateMessage_kernalILi%dELi%dEEviPtS0_PfP7ColInfoP7RowInfoPh", nCode, Size);                          
	#endif
	cutilDrvSafeCall(cuModuleGetFunction(&BPalgo->check_update, hModule, functionName));
	cuFuncSetBlockShape(BPalgo->check_update, LDPCA_BLOCK_SIZE, 1, 1);//numBlocksH[nCode], LDPCA_BLOCK_SIZE
	// set up parameter values
    offset = 0;

    cutilDrvSafeCall(cuParamSeti(BPalgo->check_update, offset, 0));
    offset += sizeof(int);

	#ifndef LDPCA_SINGLE_BUFFER		
		offset = (offset + __alignof(void*) - 1) & ~(__alignof(void*) - 1); // adjust offset to meet alignment requirement
		cutilDrvSafeCallNoSync(cuParamSetv(BPalgo->check_update, offset, &d_Vmessage, sizeof(void*)));	
		offset += sizeof(void*);
	#endif

	offset = (offset + __alignof(void*) - 1) & ~(__alignof(void*) - 1); // adjust offset to meet alignment requirement
    cutilDrvSafeCallNoSync(cuParamSetv(BPalgo->check_update, offset, &d_Hmessage, sizeof(void*)));	
    offset += sizeof(void*);

	offset = (offset + __alignof(void*) - 1) & ~(__alignof(void*) - 1); // adjust offset to meet alignment requirement
    cutilDrvSafeCallNoSync(cuParamSetv(BPalgo->check_update, offset, &d_softInputPadding, sizeof(void*)));	
    offset += sizeof(void*);

	offset = (offset + __alignof(void*) - 1) & ~(__alignof(void*) - 1); // adjust offset to meet alignment requirement
    cutilDrvSafeCallNoSync(cuParamSetv(BPalgo->check_update, offset, &d_Vinfo[nCode], sizeof(void*)));	
    offset += sizeof(void*);

	offset = (offset + __alignof(void*) - 1) & ~(__alignof(void*) - 1); // adjust offset to meet alignment requirement
    cutilDrvSafeCallNoSync(cuParamSetv(BPalgo->check_update, offset, &d_Hinfo[nCode], sizeof(void*)));	
    offset += sizeof(void*);

	offset = (offset + __alignof(void*) - 1) & ~(__alignof(void*) - 1); // adjust offset to meet alignment requirement
    cutilDrvSafeCallNoSync(cuParamSetv(BPalgo->check_update, offset, &d_decodedInfo, sizeof(void*)));	
    offset += sizeof(void*);

	cuParamSetSize(BPalgo->check_update, offset);


//void check_kernel_RT(stream, nCode, QCIF, numBlocksH[nCode], 0, d_softInputPadding, d_Vinfo[nCode], d_Hinfo[nCode], d_decodedInfo);	
	#ifdef LDPCA_SINGLE_BUFFER
		sprintf(functionName, "_Z19CheckMessage_kernalILi%dELi%dEEviPKfP7ColInfoP7RowInfoPh", nCode, Size);
	#else
		sprintf(functionName, "_Z19CheckMessage_kernalILi%dELi%dEEviPtPKfP7ColInfoP7RowInfoPh", nCode, Size);                          
	#endif
	cutilDrvSafeCall(cuModuleGetFunction(&BPalgo->check, hModule, functionName));
	cuFuncSetBlockShape(BPalgo->check, LDPCA_BLOCK_SIZE, 1, 1);//numBlocksH[nCode], LDPCA_BLOCK_SIZE
	// set up parameter values
    offset = 0;

    cutilDrvSafeCall(cuParamSeti(BPalgo->check, offset, 0));
    offset += sizeof(int);

	#ifndef LDPCA_SINGLE_BUFFER	
		offset = (offset + __alignof(void*) - 1) & ~(__alignof(void*) - 1); // adjust offset to meet alignment requirement
		cutilDrvSafeCallNoSync(cuParamSetv(BPalgo->check, offset, &d_Hmessage, sizeof(void*)));	
		offset += sizeof(void*);		
	#endif

	offset = (offset + __alignof(void*) - 1) & ~(__alignof(void*) - 1); // adjust offset to meet alignment requirement
    cutilDrvSafeCallNoSync(cuParamSetv(BPalgo->check, offset, &d_softInputPadding, sizeof(void*)));	
    offset += sizeof(void*);

	offset = (offset + __alignof(void*) - 1) & ~(__alignof(void*) - 1); // adjust offset to meet alignment requirement
    cutilDrvSafeCallNoSync(cuParamSetv(BPalgo->check, offset, &d_Vinfo[nCode], sizeof(void*)));	
    offset += sizeof(void*);

	offset = (offset + __alignof(void*) - 1) & ~(__alignof(void*) - 1); // adjust offset to meet alignment requirement
    cutilDrvSafeCallNoSync(cuParamSetv(BPalgo->check, offset, &d_Hinfo[nCode], sizeof(void*)));	
    offset += sizeof(void*);

	offset = (offset + __alignof(void*) - 1) & ~(__alignof(void*) - 1); // adjust offset to meet alignment requirement
    cutilDrvSafeCallNoSync(cuParamSetv(BPalgo->check, offset, &d_decodedInfo, sizeof(void*)));	
    offset += sizeof(void*);

	cuParamSetSize(BPalgo->check, offset);

}



void LDPCA_CUDA_Init_Variable(){
//	TimeStamp start_t, end_t;
//	timeStart(&start_t);
//cuInit(0);
	//CUDA ladder data(CPU variables)

	int maxNumBlocksH = 0;		
	//varies across codes
	int *Hidx2tid;                 // Horizontal processing oreder of H matrix structure
	RowInfo *Hinfo;                    //[numCodes][numEdges*2]     rowPos     --     rowDeg     --     H2V     --    syndrome(1)															//  2~99 degree   2 4 8 16 32 64 128  LOG(Vtid)           0 or 1
	Index2 *HinitIdx;              //[numCodes][numEdges*2]   soft input initialization
	ColInfo *Vinfo;                    //[numCodes][numEdges*2]     softIdx    --     nextIdx    --   nextIdx
	int *decodeIdx;	

	//allocate syndrome reodering index
	syndPos = (int**)malloc(sizeof(int*)*numCodes);	

	//allocate CUDA variable in CPU memory	
	decodeIdx = (int*)malloc( sizeof(int)*codeLength );
	numBlocksH = (int*)malloc( sizeof(int)*numCodes );
	Hidx2tid = (int*)malloc(sizeof(int) * numEdges);	
	syndrome_reoder_byte = (unsigned char*)malloc(sizeof(unsigned char)*(codeLength/8 + 1));
	CUDA_SAFE_CALL(cudaHostAlloc ( &decodedInfo_pinned, sizeof(unsigned char)*(codeLength + LDPCA_DECODED_HEADER), cudaHostAllocDefault)); //cudaHostAllocWriteCombined, cudaHostAllocDefault


	//allocate CUDA variable in GPU memory	
	d_Hinfo = (RowInfo**)malloc( sizeof(RowInfo*)*numCodes );
	d_HinitIdx = (Index2**)malloc( sizeof(Index2*)*numCodes );	
	d_Vinfo = (ColInfo**)malloc( sizeof(ColInfo*)*numCodes );

	int bytesN;
	if((codeLength & 7) == 0)	//i%8 == 0
		bytesN = codeLength>>3;	
	else
		bytesN = codeLength>>3 + 1;	

	cudaMalloc((void**) &d_LR_intrinsic, sizeof(float) * codeLength);	
	CUDA_SAFE_CALL(cudaMalloc((void**) &(d_syndromeByte), sizeof(char)*bytesN ));	 
	CUDA_SAFE_CALL(cudaMalloc((void**) &(d_decodedInfo), sizeof(unsigned char)*(codeLength + LDPCA_DECODED_HEADER)));

	for(int c=0; c<numCodes; c++){  
	//====================Create H Structure Variable========================
		MtxStruct *parityH = getHstruct(c);
		int N = parityH->N;
		int M = parityH->M;

	//====================H Structure Row Re-ordering========================
		syndPos[c] = (int*)malloc(sizeof(int)*M);
		HstructReorder(parityH, syndPos[c]);

	//====================Create CUDA memory structure=================
		//先計算Hinfo要malloc多大
		int Hlen = get_Hinfo_size(parityH);
		if(c>0){
			free(Hinfo);
			free(HinitIdx);
			free(Vinfo);
		}
		Hinfo = (RowInfo*)malloc(sizeof(RowInfo) * Hlen);
		HinitIdx = (Index2*)malloc(sizeof(Index2) * Hlen);
		Vinfo = (ColInfo*)malloc(sizeof(ColInfo) * Hlen);
		memset(Hinfo,0,sizeof(RowInfo) * Hlen);
		memset(HinitIdx,0,sizeof(Index2) * Hlen);
		memset(decodeIdx,0xFF,sizeof(int)*N);
		memset(Hidx2tid,0,sizeof(int) * numEdges);
		memset(Vinfo,0,sizeof(ColInfo) * Hlen);		

		int tx=0;          //thread index within a block
		int tid=0;         //thread global index
		numBlocksH[c] = 1;  //block number		
		/*-----------------------------------
			Horizontal Processing structure
		------------------------------------*/
		int preActiveNum = 0;
		//int rowCount[100];  memset(rowCount, 0, sizeof(int)*100);
		for(int k = 0; k < M; k++){			
			int activeNum = parityH->sumRow[k];
		//	rowCount[activeNum]++;
			//printf("%d  ",activeNum);
			int pow;
			int allNum;
			allNum = get2pow(parityH->sumRow[k], &pow);
			if(tx + allNum -1  >  LDPCA_BLOCK_SIZE - 1 || get2pow(activeNum, NULL) < get2pow(preActiveNum, NULL)){   // if this block does not have sufficient space
				//change to next block in CUDA										// or this is a new row having different rowSum
				tx = 0;
				tid = numBlocksH[c]*LDPCA_BLOCK_SIZE;				
				numBlocksH[c]++;
			}
			//printf("%d-%d \n",tid,tid+activeNum-1);getchar();
			//printf("%d-%d %d\n",tx,tx+activeNum-1,pow);getchar();
			for(int row_pos = 0; row_pos < activeNum; row_pos++, tid++, tx++){	
				Hinfo[tid].rowDeg = allNum;  
				int Hidx = parityH->accumSumRow[k]+row_pos;
				Hidx2tid[Hidx] = tid;
				HinitIdx[tid].softIdx = parityH->idxInRow[k][row_pos];
				Hinfo[tid].colIdx = parityH->idxInRow[k][row_pos] + LDPCA_DECODED_HEADER;
				
				HinitIdx[tid].syndIdx = k;

				int rowIdx = parityH->idxInRow[k][row_pos];
				if(decodeIdx[rowIdx]==0xFFFFFFFF)
					decodeIdx[rowIdx] = tid;

				//Hinfo[tid] = Hinfo[tid] << 1;  //nextPow2(rowDeg)  --  syndrome				
			}
			tid += allNum - activeNum;
			tx  += allNum - activeNum;
			preActiveNum = activeNum;
		}
		//printf("c=%d Hlen = %d\n",c, numBlocksH[c]*LDPCA_BLOCK_SIZE);//	getchar();	
		//printf("=========c=%d=========\n",c);//	getchar();	
		//for(int i=0; i< 100; i++){
		//	if(rowCount[i]!=0){
		//		printf("rowSum %2d x %3d\n",i, rowCount[i]);
		//	}
		//}
		//getchar();

		if(maxNumBlocksH < numBlocksH[c])
			maxNumBlocksH = numBlocksH[c];


		/*-----------------------------------
			Vertical Processing structure
		------------------------------------*/		
		for(int k = 0; k < N; k++){  			
			int activeNum = parityH->sumCol[k];					
			//printf("%d-%d \n",tid,tid+activeNum-1);//getchar();
			//printf("%d-%d\n",tx,tx+activeNum-1);getchar();
			for(int col_pos = 0; col_pos < activeNum; col_pos++, tid++, tx++){						
				int next1,next2;
				next1 = (col_pos+1)%activeNum;
				next2 = (col_pos+2)%activeNum;

				int r,Hidx,Htid1,Htid2;
				r = parityH->idxInCol[k][col_pos];
				//printf("accumSum%d r=%d k=%d idxInRow=%p sumRow=%d\n",parityH->accumSumRow[r], r, k, parityH->idxInRow[r], parityH->sumRow[r]);
				Hidx = parityH->accumSumRow[r] + getRowPos(r, k, parityH->idxInRow[r], parityH->sumRow[r]);				
				//printf("c=%d %d\n",c,Hidx);
				tid = Hidx2tid[Hidx];

				r = parityH->idxInCol[k][next1];
				Hidx = parityH->accumSumRow[r] + getRowPos(r, k, parityH->idxInRow[r], parityH->sumRow[r]);				
				Htid1 = Hidx2tid[Hidx];

				r = parityH->idxInCol[k][next2];
				Hidx = parityH->accumSumRow[r] + getRowPos(r, k, parityH->idxInRow[r], parityH->sumRow[r]);
				Htid2 = Hidx2tid[Hidx];

				Vinfo[tid].next1 = Htid1;
				Vinfo[tid].next2 = Htid2;
			}
		}	
		freeHstruct(parityH);
				
		CUDA_SAFE_CALL(cudaMalloc((void**) &(d_Vinfo[c]), sizeof(ColInfo)*Hlen));
		CUDA_SAFE_CALL(cudaMemcpy(d_Vinfo[c], Vinfo, sizeof(ColInfo)*Hlen,cudaMemcpyHostToDevice) );

		CUDA_SAFE_CALL(cudaMalloc((void**) &(d_Hinfo[c]), sizeof(RowInfo)*Hlen));
		CUDA_SAFE_CALL(cudaMalloc((void**) &(d_HinitIdx[c]), sizeof(Index2)*Hlen));
		CUDA_SAFE_CALL(cudaMemcpy(d_Hinfo[c], Hinfo, sizeof(RowInfo)*Hlen,cudaMemcpyHostToDevice) );
		CUDA_SAFE_CALL(cudaMemcpy(d_HinitIdx[c], HinitIdx, sizeof(Index2)*Hlen,cudaMemcpyHostToDevice) );

    }// for(int c=0; c<numCodes; c++)
	int maxLen = maxNumBlocksH*LDPCA_BLOCK_SIZE;
	CUDA_SAFE_CALL(cudaMalloc((void**) &(d_Vmessage), sizeof(Msg)*maxLen));
	CUDA_SAFE_CALL(cudaMalloc((void**) &(d_Hmessage), sizeof(Msg)*maxLen));
	CUDA_SAFE_CALL(cudaMalloc((void**) &(d_VmessageBit), sizeof(char)*maxLen));  //no use

	CUDA_SAFE_CALL(cudaMalloc((void**) &(d_softInputPadding), sizeof(float)*maxLen));	
	CUDA_SAFE_CALL(cudaMalloc((void**) &(d_syndromePadding), sizeof(char)*maxLen));	
			
	free(Hidx2tid);
	free(Vinfo);
	free(Hinfo);
	free(HinitIdx);
	free(decodeIdx);

	//==========Create cuda stream============
	cutilDrvSafeCallNoSync( cuStreamCreate(&updateStream, 0));
	cutilDrvSafeCallNoSync( cuStreamCreate(&checkStream, 0));

	#ifdef LDPC_ANALYSIS
		ldpcInfo = (LdpcInfo*)malloc (sizeof(LdpcInfo)*numCodes);
		memset(ldpcInfo,0,sizeof(LdpcInfo)*numCodes);
	#endif
	
	int Size;
	if(codingOption.FrameWidth == 176){
		Size = QCIF;
	}
	else{
		Size = CIF;
	}	
	BPalgo = (CUfuncs*)malloc(sizeof(CUfuncs)*(numCodes-1));
	for(int i=0;i<(numCodes-1);i++)
		load_ldpc_funcs(&BPalgo[i], i, Size);

	#ifdef LDPCA_SINGLE_BUFFER
		#ifdef RUNTIME_API
			bind_txture_RT(d_Hmessage, maxLen * sizeof(Msg));
		#else
			size_t bias;
			cuModuleGetTexRef(&hTexRef, hModule, "tex_message");
			cuTexRefSetAddress(&bias, hTexRef, (CUdeviceptr)d_Hmessage, maxLen * sizeof(Msg)) ;
			cuTexRefSetAddressMode(hTexRef, 0, CU_TR_ADDRESS_MODE_CLAMP);
			cuTexRefSetFlags(hTexRef, CU_TRSF_READ_AS_INTEGER);
			cuTexRefSetFormat(hTexRef, CU_AD_FORMAT_UNSIGNED_INT16, 1);//CU_AD_FORMAT_HALF  CU_AD_FORMAT_FLOAT  CU_AD_FORMAT_UNSIGNED_INT16
		#endif
	#endif
	
	//LDPCA_CUDA_Free_Buffer(); //test
//	printf("=======myReadLadder_decoder_CUDA======== %lfms,\n", timeElapse(start_t,&end_t));

	//cudaThreadSynchronize();
	//printf("finish!\n");
	//getchar();
}


void LDPCA_CUDA_Free_Variable(){
	
	CUDA_SAFE_CALL(cudaFree(d_Hmessage));
	CUDA_SAFE_CALL(cudaFree(d_Vmessage));
	CUDA_SAFE_CALL(cudaFree(d_VmessageBit));	
	CUDA_SAFE_CALL(cudaFree(d_syndromeByte));		
	cudaFree(d_LR_intrinsic);
	CUDA_SAFE_CALL(cudaFree(d_decodedInfo));	
    for(int i=0; i<numCodes;i++){
		free(syndPos[i]);
        CUDA_SAFE_CALL(cudaFree(d_Vinfo[i]));
		CUDA_SAFE_CALL(cudaFree(d_Hinfo[i]));
		CUDA_SAFE_CALL(cudaFree(d_HinitIdx[i]));
    }
	free(syndPos);
	CUDA_SAFE_CALL(cudaFree(d_softInputPadding));
	CUDA_SAFE_CALL(cudaFree(d_syndromePadding));	
    free(d_Vinfo);
	free(d_Hinfo);
	free(d_HinitIdx);
	
	free(numBlocksH);

	cudaFreeHost(decodedInfo_pinned);	
	free(syndrome_reoder_byte);
	cuStreamDestroy (updateStream);
	cuStreamDestroy (checkStream);

	free(BPalgo);
	#ifdef LDPC_ANALYSIS
		free(ldpcInfo);
	#endif
	cutilDrvSafeCall(cuCtxDetach(hContext));  //最後才能call!!
}

float ComputeCE_FERMI(){	
	float entropy;
	cudaMemcpy(&entropy, d_entropy, sizeof(float), cudaMemcpyDeviceToHost);
	return entropy/codeLength;
}

inline void inti_kernel(CUstream stream, int nCode){
	#ifdef RUNTIME_API
		init_kernel_RT(stream, nCode, QCIF, numBlocksH[nCode], d_HinitIdx[nCode], d_syndromeByte, d_softInput, d_Hinfo[nCode], d_softInputPadding, d_Hmessage);
	#else
		cuLaunchGridAsync(BPalgo[nCode].init, numBlocksH[nCode], 1, stream);
	#endif
}

inline void update_kernel(CUstream stream, int nCode){
#ifdef RUNTIME_API
	update_kernel_RT(stream, nCode, QCIF, numBlocksH[nCode], d_Hmessage, d_softInputPadding, d_Vinfo[nCode], d_Hinfo[nCode]);
#else
	CUfunction func = BPalgo[nCode].update;
	#ifdef LDPCA_SINGLE_BUFFER
		cuLaunchGridAsync(func, numBlocksH[nCode], 1, stream); 
	#else
		int offset = 0;
		cuParamSetv(func, offset, &d_Vmessage, sizeof(void*));	
		offset += sizeof(void*);
		offset = (offset + __alignof(void*) - 1) & ~(__alignof(void*) - 1); // adjust offset to meet alignment requirement
	    cuParamSetv(func, offset, &d_Hmessage, sizeof(void*));	
		cuLaunchGridAsync(func, numBlocksH[nCode], 1, stream); 
		Msg *tmp = d_Hmessage;      d_Hmessage = d_Vmessage;      d_Vmessage = tmp;  //swap buffer
	#endif
#endif
}

inline void check_update_kernel(CUstream stream, int nCode, int decodedIdx){
#ifdef RUNTIME_API
	check_update_kernel_RT(stream, nCode, QCIF, numBlocksH[nCode], decodedIdx, d_Hmessage, d_softInputPadding, d_Vinfo[nCode], d_Hinfo[nCode], d_decodedInfo);	
#else
	CUfunction func = BPalgo[nCode].check_update;
	cuParamSeti(func, 0, decodedIdx);
	#ifdef LDPCA_SINGLE_BUFFER
		cuLaunchGridAsync(func, numBlocksH[nCode], 1, stream);  
	#else
		int offset = sizeof(int);
		offset = (offset + __alignof(void*) - 1) & ~(__alignof(void*) - 1); // adjust offset to meet alignment requirement
		cuParamSetv(func, offset, &d_Vmessage, sizeof(void*));	
		offset += sizeof(void*);
		offset = (offset + __alignof(void*) - 1) & ~(__alignof(void*) - 1); // adjust offset to meet alignment requirement
	    cuParamSetv(func, offset, &d_Hmessage, sizeof(void*));	
		cuLaunchGridAsync(func, numBlocksH[nCode], 1, stream);  
		Msg *tmp = d_Hmessage;      d_Hmessage = d_Vmessage;      d_Vmessage = tmp;  //swap buffer
	#endif
#endif
}
inline void check_kernel(CUstream stream, int nCode, int decodedIdx){
#ifdef RUNTIME_API
	check_kernel_RT(stream, nCode, QCIF, numBlocksH[nCode], decodedIdx, d_softInputPadding, d_Vinfo[nCode], d_Hinfo[nCode], d_decodedInfo);	
#else
	CUfunction func = BPalgo[nCode].check;
	cuParamSeti(func, 0, decodedIdx);
	#ifdef LDPCA_SINGLE_BUFFER
		cuLaunchGrid(func, numBlocksH[nCode], 1);
	#else
		int offset = sizeof(int);
		offset = (offset + __alignof(void*) - 1) & ~(__alignof(void*) - 1); // adjust offset to meet alignment requirement
		cutilDrvSafeCallNoSync(cuParamSetv(func, offset, &d_Hmessage, sizeof(void*)));
		cuLaunchGrid(func, numBlocksH[nCode], 1);
		Msg *tmp = d_Hmessage;  d_Hmessage = d_Vmessage;  d_Vmessage = tmp;
	#endif
#endif
}

int beliefPropagation_CUDA_earlyJump(unsigned char *syndrome, unsigned char *decoded, int nCode, int request){	
	int M, M8, N;
	N = codeLength;
	M = syndLength[nCode];	//syndrome Length	
	int decodedInfoIdx = 0;

	//printf("nCode=%d   request: %d  syndromelen = %d\n",nCode, request, M );getchar();
	#ifdef LDPC_ANALYSIS
		int hType = nCode;
		ldpcInfo[hType].count++;

		TimeStamp start_t, end_t;		
		cudaThreadSynchronize();
		timeStart(&start_t);
	#endif
		
	//=========syndrome reodering==========	
	syndromeReorder(syndPos[nCode], syndrome, M, &M8);
	/* First time LDPCA decoing => Transmit soft input (LLR) to Device memory */	
	cudaMemcpy(d_syndromeByte, syndrome_reoder_byte, sizeof(unsigned char)*M8, cudaMemcpyHostToDevice);   //d_syndromeByte only

	memset(decoded, 0xFF, sizeof(unsigned char)*N);
	#ifdef _WIN32
		cuMemsetD8((CUdeviceptr)d_decodedInfo, JUMP_INFO, (N+LDPCA_DECODED_HEADER));	
	#else
		cuMemsetD8Async((CUdeviceptr)d_decodedInfo, JUMP_INFO, (N+LDPCA_DECODED_HEADER),updateStream);	
	#endif

	inti_kernel(checkStream, nCode);	//iteration #1	
	LDPC_iterations++;

	#ifdef LDPC_ANALYSIS
		cudaThreadSynchronize();
		ldpcInfo[hType].toGPU_time += timeElapse(start_t,&end_t);
		timeStart(&start_t);
	#endif

	int iter = 1;  //start from iteration #2	
	while( (iter + num_updates_per_memcpy_overlap-1) < MAX_ITERATIONS ){	
		check_update_kernel(0, nCode, decodedInfoIdx);		
		iter++;	
		LDPC_iterations++;

		cuMemcpyDtoHAsync(decodedInfo_pinned, (CUdeviceptr)d_decodedInfo, sizeof(unsigned char)*(N+LDPCA_DECODED_HEADER), checkStream);		

		for( int i=1; i<num_updates_per_memcpy_overlap; i++){
			update_kernel(updateStream, nCode);
			iter++;
			LDPC_iterations++;
		}

		cuStreamSynchronize(checkStream);
		//check negative early jump
		if(memcmp(decoded, decodedInfo_pinned+LDPCA_DECODED_HEADER, N) == 0){
			#ifdef LDPC_ANALYSIS
				cudaThreadSynchronize();
				ldpcInfo[hType].Row_time += timeElapse(start_t, &end_t);
				timeStart(&start_t);
			#endif
			return 0;
		}
		else
			memcpy(decoded, decodedInfo_pinned+LDPCA_DECODED_HEADER, N);

		//check positive early jump
		if(decodedInfo_pinned[decodedInfoIdx]==JUMP_INFO){  //success decode
			#ifdef LDPC_ANALYSIS
				cudaThreadSynchronize();
				ldpcInfo[hType].Row_time += timeElapse(start_t, &end_t);
				timeStart(&start_t);
			#endif
			return 1;
		}
		//if(decodedInfo_pinned[(decodedInfoIdx)+1]==JUMP_INFO){  //converge to wrong code
		//	#ifdef LDPC_ANALYSIS
		//		cudaThreadSynchronize();
		//		ldpcInfo[hType].Row_time += timeElapse(start_t, &end_t);
		//		timeStart(&start_t);
		//	#endif
		//	return 0;
		//}		
		decodedInfoIdx ^= INFO_SWAP;
	}	
	for(int i = iter; i< MAX_ITERATIONS; i++, LDPC_iterations++)
		update_kernel(0, nCode);
	check_kernel(0, nCode, decodedInfoIdx);  //check last iterations

	cuMemcpyDtoHAsync(decodedInfo_pinned, (CUdeviceptr)d_decodedInfo, sizeof(unsigned char)*(N+LDPCA_DECODED_HEADER), 0);
	cudaThreadSynchronize();
	memcpy(decoded, decodedInfo_pinned+LDPCA_DECODED_HEADER, sizeof(char)*(N));	
	#ifdef LDPC_ANALYSIS
		cudaThreadSynchronize();
		ldpcInfo[hType].Row_time += timeElapse(start_t, &end_t);
		timeStart(&start_t);
	#endif
	return decodedInfo_pinned[decodedInfoIdx]==JUMP_INFO? 1: 0;

}


int beliefPropagation_CUDA(unsigned char *syndrome, unsigned char *decoded, int nCode, int request){	
	LDPC_iterations += MAX_ITERATIONS;
	int M, M8, N;
	N = codeLength;
	M = syndLength[nCode];	//syndrome Length	
	int decodedInfoIdx = 0;

	//printf("nCode=%d   request: %d  syndromelen = %d\n",nCode, request, M );getchar();
	#ifdef LDPC_ANALYSIS
		int hType = nCode;
		ldpcInfo[hType].count++;

		TimeStamp start_t, end_t;		
		cudaThreadSynchronize();
		timeStart(&start_t);
	#endif
	//=========syndrome reodering==========	
	syndromeReorder(syndPos[nCode], syndrome, M, &M8);	
	cudaMemcpy(d_syndromeByte, syndrome_reoder_byte, sizeof(unsigned char)*M8, cudaMemcpyHostToDevice);   //d_syndromeByte only

	cuMemsetD8((CUdeviceptr)d_decodedInfo, JUMP_INFO, (N+LDPCA_DECODED_HEADER));
	inti_kernel(0, nCode);	//iteration #1		

	#ifdef LDPC_ANALYSIS
		cudaThreadSynchronize();
		ldpcInfo[hType].toGPU_time += timeElapse(start_t,&end_t);
		timeStart(&start_t);
	#endif

	for(int iter=2; iter <= MAX_ITERATIONS; iter++){	
		update_kernel(0, nCode);		
		//check_update_kernel(0, nCode, decodedInfoIdx);
		//decodedInfoIdx ^= INFO_SWAP;
	}	

	#ifdef LDPC_ANALYSIS
		cudaThreadSynchronize();
		ldpcInfo[hType].Row_time += timeElapse(start_t, &end_t);
		timeStart(&start_t);
	#endif

	check_kernel(0, nCode, decodedInfoIdx);  //check last iterations	

	#ifdef LDPC_ANALYSIS
		cudaThreadSynchronize();
		ldpcInfo[hType].BpCheck_time += timeElapse(start_t,&end_t);
		timeStart(&start_t);
	#endif

	cuMemcpyDtoHAsync(decodedInfo_pinned, (CUdeviceptr)d_decodedInfo, sizeof(unsigned char)*(N+LDPCA_DECODED_HEADER), 0);
	cudaThreadSynchronize();
	memcpy(decoded, decodedInfo_pinned+LDPCA_DECODED_HEADER, sizeof(char)*(N));	

	#ifdef LDPC_ANALYSIS
		cudaThreadSynchronize();
		ldpcInfo[hType].checkResult_time += timeElapse(start_t,&end_t);
	#endif
	
	return decodedInfo_pinned[decodedInfoIdx]==JUMP_INFO? 1: 0;
}



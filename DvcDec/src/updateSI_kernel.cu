#include "updateSI_kernel.h"
#include "sideInfoCreation.h"
#include "noiseModel_kernel.h"

#define THREADNUM 128	//A power of 2
//#undef FERMI

typedef struct __align__(4){
	short x;
	short y;
} Vector;

//#define SI_ANALYSIS
Vector *fBestMV, *pBestMV, *pfBestMV;


static unsigned char *d_futureFrame_half;
static unsigned char *d_pastFrame_half;

Vector *d_pfBestMV;  //fBestMV + pBestMV
unsigned char *d_reconsFrame;
unsigned char  *d_sideInfoFrame;

Vector *d_blockPos;  //JSRF
Vector *blockPos;  //JSRF

#ifndef FERMI
	cudaArray *d_futureFrame_array;
	cudaArray *d_pastFrame_array;
	cudaChannelFormatDesc channelDesc;
	texture<unsigned char, 2, cudaReadModeElementType> tex_pastFrame;
	texture<unsigned char, 2, cudaReadModeElementType> tex_futureFrame;
#endif

void ME_CUDA_Init_Buffer(){
	#ifndef FERMI
		//2D
		channelDesc = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
		CUDA_SAFE_CALL(cudaMallocArray( &d_pastFrame_array, &channelDesc, codingOption.FrameWidth, codingOption.FrameHeight, 0 ));  //invalid argument in CUDA 4.0
		CUDA_SAFE_CALL(cudaMallocArray( &d_futureFrame_array, &channelDesc, codingOption.FrameWidth, codingOption.FrameHeight, 0 ));

		// set texture parameters
		tex_pastFrame.addressMode[0] = cudaAddressModeClamp;
		tex_pastFrame.addressMode[1] = cudaAddressModeClamp;
		tex_pastFrame.filterMode = cudaFilterModePoint;  //cudaFilterModeLinear  cudaFilterModePoint
		tex_pastFrame.normalized = false;    // whether access with normalized texture coordinates

		// set texture parameters
		tex_futureFrame.addressMode[0] = cudaAddressModeClamp;
		tex_futureFrame.addressMode[1] = cudaAddressModeClamp;
		tex_futureFrame.filterMode = cudaFilterModePoint;  //cudaFilterModeLinear  cudaFilterModePoint
		tex_futureFrame.normalized = false;    // whether access with normalized texture coordinates

	#else
		cudaMalloc((void**) &d_futureFrame_half, sizeof(unsigned char) * FrameSize * 4);
		cudaMalloc((void**) &d_pastFrame_half, sizeof(unsigned char) * FrameSize * 4);	
	#endif	


	cudaMalloc((void**) &d_reconsFrame, sizeof(unsigned char) * FrameSize);
	cudaMalloc((void**) &d_pfBestMV, sizeof(Vector) * Block44Num * 2);
	fBestMV = pBestMV = pfBestMV = (Vector*)malloc(sizeof(Vector) * Block44Num * 2);



	cudaMalloc((void**) &d_blockPos, sizeof(Vector) * Block44Num);  //JSRF
	blockPos = (Vector*)malloc(sizeof(Vector) * Block44Num);

}

void ME_CUDA_Free_Buffer(){
		
	cudaFree(d_reconsFrame);
	cudaFree(d_pfBestMV);

	cudaFree(d_blockPos);  //JSRF

	#ifndef FERMI
		CUDA_SAFE_CALL(cudaFreeArray(d_futureFrame_array));
		CUDA_SAFE_CALL(cudaFreeArray(d_pastFrame_array));
	#else
		cudaFree(d_futureFrame_half);
		cudaFree(d_pastFrame_half);
	#endif

	free(pfBestMV);
	free(blockPos);
}

void ME_CUDA_Init_Variable(SIR *sir, int begin, int end){
	int searchCandidate = sir->searchCandidate;

	//copy reference frame from host to device
	#ifdef FERMI
		cudaMemcpy(d_futureFrame_half, decodedFrames[end].interpolateFrame, sizeof(unsigned char) * FrameSize * 4, cudaMemcpyHostToDevice);
		cudaMemcpy(d_pastFrame_half, decodedFrames[begin].interpolateFrame, sizeof(unsigned char) * FrameSize * 4, cudaMemcpyHostToDevice);
	#else
		//2D
		cudaMemcpyToArray( d_futureFrame_array, 0, 0, decodedFrames[end].reconsFrame, sizeof(unsigned char) * FrameSize, cudaMemcpyHostToDevice);
		cudaMemcpyToArray( d_pastFrame_array, 0, 0, decodedFrames[begin].reconsFrame, sizeof(unsigned char) * FrameSize, cudaMemcpyHostToDevice);
	#endif	

	for(int i=0; i<searchCandidate; i++){
		blockPos[i].x = sir->searchInfo[i].left;
		blockPos[i].y = sir->searchInfo[i].top;
	}
	cudaMemcpy(d_blockPos, blockPos, sizeof(Vector) * searchCandidate, cudaMemcpyHostToDevice);

}

//void ME_CUDA_Free_Variable(){
	//cudaFree(d_pBestMV);
//	cudaFree(d_searchInfo);
//}

__device__ __forceinline__ float SI_costFunc_CUDA(float sad, int dx, int dy){
	//return sad * ( 1.0 + K * sqrtf(dx * dx + dy * dy) );
	return sad * ( 1.0f + K * hypotf(dx, dy) );
}

__global__ void ME_kernel(unsigned char* d_sideInfoFrame, Vector *d_pBestMV, Vector *d_fBestMV, unsigned char *reconsFrame, unsigned char *pastFrame, unsigned char *futureFrame, Vector *blockPos, int frameWidth, int frameHeight, float intervPC, float intervFC){
	Vector pos = blockPos[blockIdx.x];
	//shared memory
	__shared__ int sBlock[BLOCKSIZE44];
	__shared__ int sPcost[THREADNUM];
	__shared__ Vector sPmv[THREADNUM];
	__shared__ int sFcost[THREADNUM];
	__shared__ Vector sFmv[THREADNUM];
		
	sPcost[threadIdx.x] = 100000000.0f;
	sFcost[threadIdx.x] = 100000000.0f;

	int top = pos.y;
	int left = pos.x;
	if(threadIdx.x < BLOCKSIZE44)
		sBlock[threadIdx.x] = reconsFrame[(top+(threadIdx.x>>2))*frameWidth + left + (threadIdx.x&3)];

	int searchTop = top - SEARCHRANGE; 
	int searchLeft = left - SEARCHRANGE;

	__syncthreads();		
	//#pragma unroll
	for(int i=threadIdx.x; i<(SEARCHRANGE*2)*(SEARCHRANGE*2); i+=THREADNUM){//65x65/128
		int y = (i >> 6) + searchTop;
		if(y < 0 || y > frameHeight-4)
			continue;
		int x = (i & (SEARCHRANGE*2-1)) + searchLeft;
		if(x < 0 || x > frameWidth-4)
			continue;
		//calculate cost of each candidate block
		//mean absolute difference (4x4) 
		#ifdef FERMI
			int rowInFrame = y * (frameWidth<<2) + (x<<1);
		#endif
		int pSad=0, fSad=0;
		
		int idxInBlk=0;
		#pragma unroll 4 
		for(int m=0; m<4; m++){
			#ifdef FERMI
				pSad = __sad(sBlock[idxInBlk], pastFrame[rowInFrame], pSad); 
				fSad = __sad(sBlock[idxInBlk++], futureFrame[rowInFrame], fSad); 
				pSad = __sad(sBlock[idxInBlk], pastFrame[rowInFrame+2], pSad); 
				fSad = __sad(sBlock[idxInBlk++], futureFrame[rowInFrame+2], fSad); 
				pSad = __sad(sBlock[idxInBlk], pastFrame[rowInFrame+4], pSad); 
				fSad = __sad(sBlock[idxInBlk++], futureFrame[rowInFrame+4], fSad); 
				pSad = __sad(sBlock[idxInBlk], pastFrame[rowInFrame+6], pSad); 
				fSad = __sad(sBlock[idxInBlk++], futureFrame[rowInFrame+6], fSad); 
				rowInFrame += (frameWidth<<2); 	
			#else			
				pSad = __sad(sBlock[idxInBlk], tex2D(tex_pastFrame, x, y+m), pSad); 
				fSad = __sad(sBlock[idxInBlk++], tex2D(tex_futureFrame, x, y+m), fSad); 
				pSad = __sad(sBlock[idxInBlk], tex2D(tex_pastFrame, x+1, y+m), pSad); 
				fSad = __sad(sBlock[idxInBlk++], tex2D(tex_futureFrame, x+1, y+m), fSad); 
				pSad = __sad(sBlock[idxInBlk], tex2D(tex_pastFrame, x+2, y+m), pSad); 
				fSad = __sad(sBlock[idxInBlk++], tex2D(tex_futureFrame, x+2, y+m), fSad); 
				pSad = __sad(sBlock[idxInBlk], tex2D(tex_pastFrame, x+3, y+m), pSad); 
				fSad = __sad(sBlock[idxInBlk++], tex2D(tex_futureFrame, x+3, y+m), fSad); 
			#endif
		} 

		float w = ( K * hypotf((x-left)<<1, (y-top)<<1) + 1.0f);
		pSad = (float)pSad* w;
		fSad = (float)fSad* w;

		if(pSad < sPcost[threadIdx.x]){
			sPcost[threadIdx.x] = pSad;
			sPmv[threadIdx.x].x = x;
			sPmv[threadIdx.x].y = y;
		}
		if(fSad < sFcost[threadIdx.x]){
			sFcost[threadIdx.x] = fSad;
			sFmv[threadIdx.x].x = x;
			sFmv[threadIdx.x].y = y;
		}

	}

	//find best motion vector
	int candidate = THREADNUM;
	while(candidate > 1){
		__syncthreads();
		candidate >>= 1;
		if(threadIdx.x < candidate){
			if(sPcost[threadIdx.x] > sPcost[threadIdx.x + candidate]){
				sPcost[threadIdx.x] = sPcost[threadIdx.x + candidate];
				sPmv[threadIdx.x] = sPmv[threadIdx.x + candidate];
			}
			if(sFcost[threadIdx.x] > sFcost[threadIdx.x + candidate]){
				sFcost[threadIdx.x] = sFcost[threadIdx.x + candidate];
				sFmv[threadIdx.x] = sFmv[threadIdx.x + candidate];
			}
		}
	}	
	#ifndef FERMI
		if(threadIdx.x == 0){
			d_pBestMV[blockIdx.x] = sPmv[0];
			d_fBestMV[blockIdx.x] = sFmv[0];
		}
	#else
		//=========initial==========
		sPcost[threadIdx.x] = 100000000.0f;
		sFcost[threadIdx.x] = 100000000.0f;
		//==========SubPel ME==============
		if(threadIdx.x < 9){  //past ME
			Vector subIdx[9] = { {-1, -1}, {0, -1}, {1, -1}, {-1, 0}, {0, 0}, {1, 0}, {-1, 1}, {0, 1}, {1, 1} };
			int x,y;
			y = (sPmv[0].y<<1) + subIdx[threadIdx.x].y;
			x = (sPmv[0].x<<1) + subIdx[threadIdx.x].x;

			if( y >= 0 && x >= 0 && x <= (frameWidth<<1)-8 && y <= (frameHeight<<1)-8){
				//calculate cost of each candidate block
				//mean absolute difference (4x4) 
				int rowInFrame = y * (frameWidth<<1) + x;
				int pSad=0;
				
				int idxInBlk=0;
				#pragma unroll 4 
				for(int m=0; m<4; m++){
					pSad = __sad(sBlock[idxInBlk++], pastFrame[rowInFrame], pSad); 
					pSad = __sad(sBlock[idxInBlk++], pastFrame[rowInFrame+2], pSad); 
					pSad = __sad(sBlock[idxInBlk++], pastFrame[rowInFrame+4], pSad); 
					pSad = __sad(sBlock[idxInBlk++], pastFrame[rowInFrame+6], pSad); 
					rowInFrame += (frameWidth<<2); 
				} 
				//float w = ( K * hypotf(x-(left<<1), y-(top<<1)) + 1.0f);
				sPcost[threadIdx.x] = pSad;
			}
			sPmv[threadIdx.x].x = x;
			sPmv[threadIdx.x].y = y;

			y = (sFmv[0].y<<1) + subIdx[threadIdx.x].y;
			x = (sFmv[0].x<<1) + subIdx[threadIdx.x].x;
			if( y >= 0 && x >= 0 && x <= (frameWidth<<1)-8 && y <= (frameHeight<<1)-8){			
				//calculate cost of each candidate block
				//mean absolute difference (4x4) 
				int rowInFrame = y * (frameWidth<<1) + x;
				int fSad=0;
				
				int idxInBlk=0;
				#pragma unroll 4 
				for(int m=0; m<4; m++){
					fSad = __sad(sBlock[idxInBlk++], futureFrame[rowInFrame], fSad); 
					fSad = __sad(sBlock[idxInBlk++], futureFrame[rowInFrame+2], fSad); 
					fSad = __sad(sBlock[idxInBlk++], futureFrame[rowInFrame+4], fSad); 
					fSad = __sad(sBlock[idxInBlk++], futureFrame[rowInFrame+6], fSad); 
					rowInFrame += (frameWidth<<2); 
				} 
				//float w = ( K * hypotf(x-(left<<1), y-(top<<1)) + 1.0f);
				sFcost[threadIdx.x] = fSad;
			}
			sFmv[threadIdx.x].x = x;
			sFmv[threadIdx.x].y = y;
		}
		//find best motion vector
		candidate = 16;
		while(candidate > 1){
			__syncthreads();
			candidate >>= 1;
			if(threadIdx.x < candidate){
				if(sFcost[threadIdx.x] > sFcost[threadIdx.x + candidate]){
					sFcost[threadIdx.x] = sFcost[threadIdx.x + candidate];
					sFmv[threadIdx.x] = sFmv[threadIdx.x + candidate];
				}
				if(sPcost[threadIdx.x] > sPcost[threadIdx.x + candidate]){
					sPcost[threadIdx.x] = sPcost[threadIdx.x + candidate];
					sPmv[threadIdx.x] = sPmv[threadIdx.x + candidate];
				}
			}
		}	
		if(threadIdx.x < BLOCKSIZE44){
			d_sideInfoFrame[(top+(threadIdx.x>>2))*frameWidth + left + (threadIdx.x&3)] = intervPC * pastFrame[(sPmv[0].y+((threadIdx.x>>2)<<1)) * (frameWidth<<1) + (sPmv[0].x + ((threadIdx.x&3)<<1))]
																						+ intervFC * futureFrame[(sFmv[0].y+((threadIdx.x>>2)<<1)) * (frameWidth<<1) + (sFmv[0].x + ((threadIdx.x&3)<<1))];								
		}
	#endif
}

void updateSI_CUDA(SideInfo *si, unsigned char *reconsFrame, int current, int past, int future, SIR *sir){
	#ifdef SI_ANALYSIS
		TimeStamp start_t, end_t;
		double GPU_time=0, CPU_time=0;
		cudaThreadSynchronize();
		timeStart(&start_t);
	#endif

	int width = codingOption.FrameWidth;
	int height = codingOption.FrameHeight;
	float intervFC = (float)(future-current) / (float)(future-past);
	float intervPC = (float)(current-past) / (float)(future-past);

	//bidirectional motion estimation	
	dim3 Grids(sir->searchCandidate,1,1);
	dim3 Blocks(THREADNUM,1,1);
	cudaMemcpy(d_reconsFrame, reconsFrame, sizeof(unsigned char) * FrameSize, cudaMemcpyHostToDevice);
	#ifdef FERMI
		cudaMemcpy(d_sideInfoFrame, si->sideInfoFrame, sizeof(unsigned char) * FrameSize, cudaMemcpyHostToDevice);
	#else
		//2D
		cudaBindTextureToArray(tex_pastFrame, d_pastFrame_array, channelDesc);
		cudaBindTextureToArray(tex_futureFrame, d_futureFrame_array, channelDesc);
	#endif
		ME_kernel<<<Grids, Blocks>>>(d_sideInfoFrame, d_pfBestMV, d_pfBestMV+sir->searchCandidate, d_reconsFrame, d_pastFrame_half, d_futureFrame_half, d_blockPos, width, height, intervPC, intervFC); 
	#ifdef SI_ANALYSIS
		cudaThreadSynchronize();
		GPU_time = timeElapse(start_t,&end_t);		
	#endif
	#ifndef FERMI
		unsigned char *futureFrame_half = decodedFrames[future].interpolateFrame;
		unsigned char *pastFrame_half = decodedFrames[past].interpolateFrame;
		int width2 = width << 1;
		int height2 = height << 1;

		cudaUnbindTexture(tex_pastFrame);
		cudaUnbindTexture(tex_futureFrame);
		cudaMemcpy(pfBestMV, d_pfBestMV, sizeof(Vector) * sir->searchCandidate * 2, cudaMemcpyDeviceToHost);	
		pBestMV = pfBestMV;
		fBestMV = pfBestMV + sir->searchCandidate;
		//half pel refinement and motion compensation
		#pragma omp parallel for
		for(int i=0; i<sir->searchCandidate; i++){

			SearchInfo searchInfo = sir->searchInfo[i];
			int top = searchInfo.top;
			int left = searchInfo.left;		
			int pBestX = pBestMV[i].x;
			int pBestY = pBestMV[i].y;
			int fBestX = fBestMV[i].x;
			int fBestY = fBestMV[i].y;

			double block[BLOCKSIZE44];
			int idxInBlk = 0;
			int tmp = top * width + left;
			int rowInFrame = tmp;
			for(int m=0; m<4; m++){
				block[idxInBlk++] = (double)reconsFrame[rowInFrame];
				block[idxInBlk++] = (double)reconsFrame[rowInFrame + 1];
				block[idxInBlk++] = (double)reconsFrame[rowInFrame + 2];
				block[idxInBlk++] = (double)reconsFrame[rowInFrame + 3];
				rowInFrame += width;
			}

			subPelRefine(block, pastFrame_half, &pBestX, &pBestY, width2, height2);
			subPelRefine(block, futureFrame_half, &fBestX, &fBestY, width2, height2);

			int width4 = width2<<1;
			rowInFrame = tmp;
			int pRow = pBestY * width2 + pBestX;
			int fRow = fBestY * width2 + fBestX;
			for(int m=0; m<8; m+=2){
				si->sideInfoFrame[rowInFrame] = (int)((double)pastFrame_half[pRow] * intervPC + (double)futureFrame_half[fRow] * intervFC);
				si->sideInfoFrame[rowInFrame+1] = (int)((double)pastFrame_half[pRow + 2] * intervPC + (double)futureFrame_half[fRow + 2] * intervFC);
				si->sideInfoFrame[rowInFrame+2] = (int)((double)pastFrame_half[pRow + 4] * intervPC + (double)futureFrame_half[fRow + 4] * intervFC);
				si->sideInfoFrame[rowInFrame+3] = (int)((double)pastFrame_half[pRow + 6] * intervPC + (double)futureFrame_half[fRow + 6] * intervFC);
				rowInFrame += width;
				pRow += width4;
				fRow += width4;
			}

		}
		cudaMemcpy(d_sideInfoFrame, si->sideInfoFrame, sizeof(unsigned char) * FrameSize, cudaMemcpyHostToDevice);
	#else
		cudaMemcpy(si->sideInfoFrame, d_sideInfoFrame, sizeof(unsigned char) * FrameSize, cudaMemcpyDeviceToHost);
	#endif
	#ifdef SI_ANALYSIS
		cudaThreadSynchronize();
		double all = timeElapse(start_t,&end_t);
		CPU_time = all - GPU_time;
		printf("gpu:%.2lf%%\tcpu:%.2lf%%",100*GPU_time/all, 100*CPU_time/all);getchar();
	#endif
}


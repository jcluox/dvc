#include "motionLearning_kernel.h"
#include "noiseModel_kernel.h"
#include "updateSI_kernel.h"

//#include <windows.h>

#define THREADNUM 32	//A power of 2


texture<unsigned char , 1, cudaReadModeElementType> tex_refFrame;

void updateSMF_CUDA_Init_Buffer(){
	cudaMalloc((void**) &d_sideInfoFrame, sizeof(unsigned char ) * FrameSize);
}

void updateSMF_CUDA_Free_Buffer(){
	cudaFree(d_sideInfoFrame);
}

__global__ void updateSMF_kernel(SMF *d_smf, unsigned char *d_reconsFrame, unsigned char* d_sideInfoFrame, SearchInfo *d_searchInfo, int frameWidth){
	//int top = d_searchInfo[blockIdx.x].top;
	//int left = d_searchInfo[blockIdx.x].left;

	SMF blkSMF = d_smf[blockIdx.x];

	int rowInFrame;
	int SSE;
	int blkIdx;
	int diff ;

	__shared__ int sBlock[BLOCKSIZE44];
	__shared__ float sProb[ML_RANGE];
	//__shared__ double sSum[THREADNUM];
	__shared__ float sSum[THREADNUM];
	sSum[threadIdx.x] = 0.0f;
	
	if(threadIdx.x < BLOCKSIZE44){
		sBlock[threadIdx.x] = d_reconsFrame[(d_searchInfo[blockIdx.x].top+(threadIdx.x>>2))*frameWidth + d_searchInfo[blockIdx.x].left + (threadIdx.x&3)];
	}
	__syncthreads();

	for(int i=threadIdx.x; i<blkSMF.probLen; i+=THREADNUM){
		rowInFrame = (blkSMF.searchTop + i/blkSMF.searchWidth) * frameWidth + blkSMF.searchLeft + i%blkSMF.searchWidth;
		SSE = 0;
		blkIdx = 0;
		#pragma unroll 4 
		for(int m=0; m<4; m++){
			#ifdef FERMI
				diff = sBlock[blkIdx++] - d_sideInfoFrame[rowInFrame];	SSE += diff * diff;
				diff = sBlock[blkIdx++] - d_sideInfoFrame[rowInFrame+1];	SSE += diff * diff;
				diff = sBlock[blkIdx++] - d_sideInfoFrame[rowInFrame+2];	SSE += diff * diff;
				diff = sBlock[blkIdx++] - d_sideInfoFrame[rowInFrame+3];	SSE += diff * diff;
			#else
				diff = sBlock[blkIdx++] - tex1Dfetch(tex_refFrame, rowInFrame);	SSE += diff * diff;
				diff = sBlock[blkIdx++] - tex1Dfetch(tex_refFrame, rowInFrame + 1);	SSE += diff * diff;
				diff = sBlock[blkIdx++] - tex1Dfetch(tex_refFrame, rowInFrame + 2);	SSE += diff * diff;
				diff = sBlock[blkIdx++] - tex1Dfetch(tex_refFrame, rowInFrame + 3);	SSE += diff * diff;
			#endif
			rowInFrame += frameWidth;
		}
		sProb[i] = __expf(-MU * (float)SSE);
		sSum[threadIdx.x] += sProb[i];
	}

	if(threadIdx.x < 16)
		sSum[threadIdx.x] += sSum[threadIdx.x + 16];
	if(threadIdx.x < 8)
		sSum[threadIdx.x] += sSum[threadIdx.x + 8];
	if(threadIdx.x < 4)
		sSum[threadIdx.x] += sSum[threadIdx.x + 4];
	if(threadIdx.x < 2)
		sSum[threadIdx.x] += sSum[threadIdx.x + 2];
	if(threadIdx.x < 1)
		sSum[threadIdx.x] += sSum[threadIdx.x + 1];
	__syncthreads();

	if(sSum[0] > 0.0f){
		//normalize
		for(int i=threadIdx.x; i<blkSMF.probLen; i+=THREADNUM){
			//d_smf[blockIdx.x].prob[i] = (float)((double)sProb[i] / sSum[0]);
			d_smf[blockIdx.x].prob[i] = sProb[i] / sSum[0];
		}
	}
	else if(threadIdx.x == 0){
		d_smf[blockIdx.x].prob[blkSMF.blkIdx] = 1.0f;
	}

}


void updateSMF_CUDA(SMF *smf, unsigned char *reconsFrame, unsigned char  *sideInfoFrame, SIR *sir){
//cudaThreadSynchronize();
	dim3 Grids(sir->searchCandidate,1,1);
	dim3 Blocks(THREADNUM,1,1);
	//cudaMemcpy(d_sideInfoFrame, sideInfoFrame, sizeof(unsigned char) * FrameSize, cudaMemcpyHostToDevice);

	#ifndef FERMI
		cudaBindTexture(0, tex_refFrame, d_sideInfoFrame);
	#endif
	updateSMF_kernel<<<Grids, Blocks>>>(d_smf, d_reconsFrame, d_sideInfoFrame, d_searchInfo, codingOption.FrameWidth);
	#ifndef FERMI
		cudaUnbindTexture(tex_refFrame);
	#endif

	cudaMemcpy(smf, d_smf, sizeof(SMF) * sir->searchCandidate, cudaMemcpyDeviceToHost);
//cudaThreadSynchronize();
}

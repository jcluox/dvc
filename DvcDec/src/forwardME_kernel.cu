#include "forwardME_kernel.h"
#include "sideInfoCreation.h"

//#include "windows.h"

#define THREADNUM 32 //for GET_FMV_kernel (A power of 2)


typedef struct __align__(8){
	float x;
	float y;
} Vector;

typedef struct __align__(16){
	Vector vector;
	Vector hole;
} MacroMV;

texture<unsigned char, 1, cudaReadModeElementType> tex_refFrame;

MacroMV *macroBlockMV;
MacroMV *d_macroBlockMV;
MacroMV *d_si_macroBlockMV;
//SearchInfo *d_searchInfo1616;
static unsigned char *d_futureFrame, *d_pastFrame;

void ForwardME_CUDA_Init_Buffer(){
	macroBlockMV = (MacroMV*)malloc(sizeof(MacroMV) * Block1616Num);

	cudaMalloc((void**) &d_macroBlockMV, sizeof(MacroMV) * Block1616Num);
	cudaMalloc((void**) &d_futureFrame, sizeof(unsigned char) * FrameSize);
	cudaMalloc((void**) &d_pastFrame, sizeof(unsigned char) * FrameSize);
	cudaMalloc((void**) &d_si_macroBlockMV, sizeof(MacroMV) * Block1616Num);
	//cudaMalloc((void**) &d_searchInfo1616, sizeof(SearchInfo) * Block1616Num);
	//cudaMemcpy(d_searchInfo1616, SearchInfo1616, sizeof(SearchInfo) * Block1616Num, cudaMemcpyHostToDevice);
}

void ForwardME_CUDA_Free_Buffer(){
	free(macroBlockMV);

	cudaFree(d_macroBlockMV);
	//cudaFree(d_searchInfo1616);
	cudaFree(d_futureFrame);
	cudaFree(d_pastFrame);
	cudaFree(d_si_macroBlockMV);
}


__device__ float costFunc_CUDA(float mad, int dx, int dy){
	//return mad * ( 1.0f + K * sqrtf(dx * dx + dy * dy) );
	return mad * ( 1.0f + K * hypotf(dx, dy) );
}
 

__global__ void FME_kernel(MacroMV *d_macroBlockMV, unsigned char *d_pastFrame, unsigned char *futureFrame, float intervPC, float intervFC){
	int tid = threadIdx.x, bid = blockIdx.y * gridDim.x + blockIdx.x;	

	//shared memory
	__shared__ int sBlock[BLOCKSIZE1616];
	__shared__ float sCost[BLOCKSIZE1616];
	__shared__ Vector sPoint[BLOCKSIZE1616];
	
	int top = blockIdx.y<<4;
	int left = blockIdx.x<<4;

	int searchTop = top - SEARCHRANGE; 
	int searchLeft = left - SEARCHRANGE;

	
	sBlock[tid] = futureFrame[(top+(tid>>4))*(gridDim.x<<4) + left + (tid&15)];
	float left_intervPC = (float)left * intervPC;
	float top_intervPC = (float)top * intervPC;
	sCost[tid] = 10000000.0f;
	__syncthreads();
	for(int i=tid; i<(SEARCHRANGE*2)*(SEARCHRANGE*2); i+=BLOCKSIZE1616){	
		int y = (i >> 6) + searchTop;
		if(y < 0 || y > (gridDim.y<<4)-16)
			continue;
		int x = (i & (SEARCHRANGE*2-1)) + searchLeft;
		if(x < 0 || x > (gridDim.x<<4)-16)
			continue;
		//calculate cost of each candidate block
		int sad = 0;
		int idxInBlk = 0;
		int rowInFrame = y * (gridDim.x<<4) + x;
		#pragma unroll 16
		for(int m=0; m<16; m++){
			#ifdef FERMI
				sad = __sad(sBlock[idxInBlk++], d_pastFrame[rowInFrame  ], sad); sad = __sad(sBlock[idxInBlk++],  d_pastFrame[rowInFrame+1], sad);
				sad = __sad(sBlock[idxInBlk++], d_pastFrame[rowInFrame+2], sad); sad = __sad(sBlock[idxInBlk++],  d_pastFrame[rowInFrame+3], sad);
				sad = __sad(sBlock[idxInBlk++], d_pastFrame[rowInFrame+4], sad); sad = __sad(sBlock[idxInBlk++],  d_pastFrame[rowInFrame+5], sad);
				sad = __sad(sBlock[idxInBlk++], d_pastFrame[rowInFrame+6], sad); sad = __sad(sBlock[idxInBlk++],  d_pastFrame[rowInFrame+7], sad);
				sad = __sad(sBlock[idxInBlk++], d_pastFrame[rowInFrame+8], sad); sad = __sad(sBlock[idxInBlk++],  d_pastFrame[rowInFrame+9], sad);
				sad = __sad(sBlock[idxInBlk++], d_pastFrame[rowInFrame+10], sad); sad = __sad(sBlock[idxInBlk++],  d_pastFrame[rowInFrame+11], sad);
				sad = __sad(sBlock[idxInBlk++], d_pastFrame[rowInFrame+12], sad); sad = __sad(sBlock[idxInBlk++],  d_pastFrame[rowInFrame+13], sad);
				sad = __sad(sBlock[idxInBlk++], d_pastFrame[rowInFrame+14], sad); sad = __sad(sBlock[idxInBlk++],  d_pastFrame[rowInFrame+15], sad);
				rowInFrame += (gridDim.x<<4);
			#else
				sad = __sad(sBlock[idxInBlk++], tex1Dfetch(tex_refFrame, rowInFrame), sad); sad = __sad(sBlock[idxInBlk++], tex1Dfetch(tex_refFrame, rowInFrame + 1), sad);
				sad = __sad(sBlock[idxInBlk++], tex1Dfetch(tex_refFrame, rowInFrame + 2), sad); sad = __sad(sBlock[idxInBlk++], tex1Dfetch(tex_refFrame, rowInFrame + 3), sad);
				sad = __sad(sBlock[idxInBlk++], tex1Dfetch(tex_refFrame, rowInFrame + 4), sad); sad = __sad(sBlock[idxInBlk++], tex1Dfetch(tex_refFrame, rowInFrame + 5), sad);
				sad = __sad(sBlock[idxInBlk++], tex1Dfetch(tex_refFrame, rowInFrame + 6), sad); sad = __sad(sBlock[idxInBlk++], tex1Dfetch(tex_refFrame, rowInFrame + 7), sad);
				sad = __sad(sBlock[idxInBlk++], tex1Dfetch(tex_refFrame, rowInFrame + 8), sad); sad = __sad(sBlock[idxInBlk++], tex1Dfetch(tex_refFrame, rowInFrame + 9), sad);
				sad = __sad(sBlock[idxInBlk++], tex1Dfetch(tex_refFrame, rowInFrame + 10), sad); sad = __sad(sBlock[idxInBlk++], tex1Dfetch(tex_refFrame, rowInFrame + 11), sad);
				sad = __sad(sBlock[idxInBlk++], tex1Dfetch(tex_refFrame, rowInFrame + 12), sad); sad = __sad(sBlock[idxInBlk++], tex1Dfetch(tex_refFrame, rowInFrame + 13), sad);
				sad = __sad(sBlock[idxInBlk++], tex1Dfetch(tex_refFrame, rowInFrame + 14), sad); sad = __sad(sBlock[idxInBlk++], tex1Dfetch(tex_refFrame, rowInFrame + 15), sad);
				rowInFrame += (gridDim.x<<4);
			#endif
		}
		//sCost[tid] = costFunc_CUDA((float)sad, x-left, y-top);
		sad = costFunc_CUDA((float)sad, x-left, y-top);
		if(sad < sCost[tid]){
			sCost[tid] = sad;
			sPoint[tid].y = (float)y;
			sPoint[tid].x = (float)x;
		}		
	}

	//find best motion vector
	int candidate = BLOCKSIZE1616;
	while(candidate > 1){
		__syncthreads();
		candidate >>= 1;
		if(tid < candidate){
			int tmpid = tid + candidate;
			if(sCost[tid] > sCost[tmpid]){
				sCost[tid] = sCost[tmpid];
				sPoint[tid] = sPoint[tmpid];
			}
		}
	}

	if(tid == 0){
		float px = sPoint[0].x;
		float py = sPoint[0].y;
		MacroMV mbMV;
		mbMV.hole.x = left_intervPC + px * intervFC;
		mbMV.hole.y = top_intervPC + py * intervFC;
		mbMV.vector.x = px - (float)left;
		mbMV.vector.y = py - (float)top;
		d_macroBlockMV[bid] = mbMV;
	}
}

__global__ void GET_FMV_kernel(MacroMV *d_si_macroBlockMV, MacroMV *d_macroBlockMV, float maxDst){
	int tid = threadIdx.x, bid = blockIdx.y * gridDim.x + blockIdx.x;
	int block1616Num = gridDim.x * gridDim.y;	
	float top = blockIdx.y<<4;
	float left = blockIdx.x<<4;

	//shared memory
	__shared__ float sDst[THREADNUM];
	__shared__ MacroMV sMV[THREADNUM];
	sDst[tid] = maxDst;
	__syncthreads();

	for(int i=tid; i<block1616Num; i+=THREADNUM){
		MacroMV mv = d_macroBlockMV[i];
		float dx = left - mv.hole.x;
		float dy = top - mv.hole.y; 
		float dst = dx*dx + dy*dy;
		if(dst < sDst[tid]){
			sDst[tid] = dst;
			sMV[tid] = mv;
		}
	}

	int candidate = THREADNUM;
	while(candidate > 1){
		__syncthreads();
		candidate >>= 1;
		if(tid < candidate){
			int tmpid = tid + candidate;
			if(sDst[tid] > sDst[tmpid]){
				sDst[tid] = sDst[tmpid];
				sMV[tid] = sMV[tmpid];
			}
		}
	}

	if(tid == 0){
		d_si_macroBlockMV[bid] = sMV[0];
	}
}


void forwardME_CUDA(int future, int past, int current, SideInfo *si){

	int searchRangeBottom = codingOption.FrameHeight - 16;
	int searchRangeRight = codingOption.FrameWidth - 16;
	float intervFC = (float)(future-current) / (float)(future-past);
	float intervPC = (float)(current-past) / (float)(future-past);

//	cudaThreadSynchronize();
//	TimeStamp start_t, end_t;
//	timeStart(&start_t);

	//forward motion estimation
	dim3 Grids(Block1616Width, Block1616Height,1);
	dim3 Blocks(BLOCKSIZE1616,1,1);
	cudaMemcpy(d_futureFrame, decodedFrames[future].lowPassFrame, sizeof(unsigned char) * FrameSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pastFrame, decodedFrames[past].lowPassFrame, sizeof(unsigned char) * FrameSize, cudaMemcpyHostToDevice);
	#ifndef FERMI
		cudaBindTexture(0, tex_refFrame, d_pastFrame);
	#endif
	FME_kernel<<<Grids, Blocks>>>(d_macroBlockMV, d_pastFrame, d_futureFrame, intervPC, intervFC);
	#ifndef FERMI
		cudaUnbindTexture(tex_refFrame);
	#endif
	
	//cudaThreadSynchronize();
	//getchar();
//	printf("CUDA = %lfms \n",timeElapse(start_t,&end_t));

	//find motion vector of each macroblock for WZ frame
	int maxDst = codingOption.FrameWidth * codingOption.FrameWidth + codingOption.FrameHeight * codingOption.FrameHeight;

	GET_FMV_kernel<<<Grids, THREADNUM>>>(d_si_macroBlockMV, d_macroBlockMV, (float)maxDst);
	cudaMemcpy(macroBlockMV, d_si_macroBlockMV, sizeof(MacroMV) * Block1616Num, cudaMemcpyDeviceToHost);

	for(int i=0; i<Block1616Num; i++){
		int top = SearchInfo1616[i].top, left = SearchInfo1616[i].left;
		MacroMV bestMV = macroBlockMV[i];
		MV *mbMV = &(si->mcroBlockMV[i]);
		mbMV->Future.x = (double)computeEndPoint(left, -bestMV.vector.x, intervFC, 0, searchRangeRight);
		mbMV->Future.y = (double)computeEndPoint(top, -bestMV.vector.y, intervFC, 0, searchRangeBottom);
		mbMV->Past.x = (double)computeEndPoint(left, bestMV.vector.x, intervPC, 0, searchRangeRight);
		mbMV->Past.y = (double)computeEndPoint(top, bestMV.vector.y, intervPC, 0, searchRangeBottom);
		mbMV->vector.x = mbMV->Past.x - mbMV->Future.x;
		mbMV->vector.y = mbMV->Past.y - mbMV->Future.y;
	}
	//QueryPerformanceCounter(&end);
	//printf("time2: %lfms\n",(double)(end.QuadPart-st.QuadPart)*1000.0/CPUFreq);
}
#include "noiseModel_kernel.h"
#include "LDPCA_cuda.h"
#include "reconstruction.h"

#define THREADNUM1 96	//for CNM distribution
#define THREADNUM2 32	//for CNM conditional prob (A power of 2)

static float *d_distribution;

//CNM distribution
extern __shared__ int shared[];
float *alphaCeff, *d_alphaCeff;
int *sideCeff_DC, *d_sideCeff_DC;
int *sideCeff_AC, *d_sideCeff_AC;
SMF *d_smf;
float *d_entropy;
//float *d_smf_prob;

//CNM conditional prob
int *reconsCoeff, *d_reconsCoeff;
//static SearchInfo *d_searchInfo;
SearchInfo *d_searchInfo;
void CNM_CUDA_Init_Buffer(){
	//CNM distribution	
	int sideCeffSize = Block44Num * ML_RANGE;
	alphaCeff = (float*)malloc(sizeof(float) * Block44Num);
	sideCeff_DC = (int*)malloc(sizeof(int) * Block44Num);
	sideCeff_AC = (int*)malloc(sizeof(int) * sideCeffSize);
	cudaMalloc((void**) &d_alphaCeff, sizeof(float) * Block44Num);
	cudaMalloc((void**) &d_sideCeff_DC, sizeof(int) * Block44Num);
	cudaMalloc((void**) &d_sideCeff_AC, sizeof(int) * sideCeffSize);
	cudaMalloc((void**) &d_smf, sizeof(SMF) * Block44Num);
	//cudaMalloc((void**) &d_smf_prob, sizeof(float) * sideCeffSize);
	cudaMalloc((void**) &d_distribution, sizeof(float) * Block44Num * 1024);
	
	//CNM conditional prob
	reconsCoeff = (int*)malloc(sizeof(int) * Block44Num);	
	cudaMalloc((void**) &d_reconsCoeff, sizeof(int) * Block44Num);
	cudaMalloc((void**) &d_searchInfo, sizeof(SearchInfo) * Block44Num);
	cudaMalloc((void**) &d_entropy, sizeof(float));
}

void CNM_CUDA_Free_Buffer(){
	//CNM distribution	
	cudaFree(d_entropy);
	cudaFree(d_alphaCeff);
	cudaFree(d_sideCeff_DC);
	cudaFree(d_sideCeff_AC);
	free(alphaCeff);
	free(sideCeff_DC);
	free(sideCeff_AC);
	cudaFree(d_smf);

	//CNM conditional prob
	cudaFree(d_reconsCoeff);
	cudaFree(d_searchInfo);
	cudaFree(d_distribution);
	free(reconsCoeff);
}

void CNM_CUDA_Init_Variable(SIR *sir, SMF *smf){
	//CNM distribution
	int searchCandidate = sir->searchCandidate;
	cudaMemcpy(d_smf, smf, sizeof(SMF) * searchCandidate, cudaMemcpyHostToDevice);
	//cudaMemcpy(d_smf_prob, smf, sizeof(SMF) * searchCandidate, cudaMemcpyHostToDevice);
	
	
	//CNM conditional prob
	cudaMemcpy(d_searchInfo, sir->searchInfo, sizeof(SearchInfo) * searchCandidate, cudaMemcpyHostToDevice);
	//printf("searchCandidate=%d  \n",searchCandidate); 
	//for(int i = 0; i<searchCandidate; i++){		
	//	printf("blkIdx %d\n",sir->searchInfo[i].blkIdx);
	//	printf("left %d\n",sir->searchInfo[i].left);
	//	printf("searchBottom %d\n",sir->searchInfo[i].searchBottom);
	//	printf("searchLeft %d\n",sir->searchInfo[i].searchLeft);
	//	printf("searchRange %d\n",sir->searchInfo[i].searchRange);
	//	printf("searchRight %d\n",sir->searchInfo[i].searchRight);
	//	printf("searchTop %d\n",sir->searchInfo[i].searchTop);
	//	printf("searchWidth %d\n",sir->searchInfo[i].searchWidth);
	//	printf("top %d\n\n",sir->searchInfo[i].top);getchar();
	//}getchar();
}

void init_noiseDst_CUDA(int range, int indexInBlk, int searchCandidate){
	//if(indexInBlk == 0){
	//	//DC		
	//	int distrSize = searchCandidate * range;
	//	cudaMalloc((void**) &d_distribution, sizeof(float) * distrSize);
	//}
	//else{
	//	//AC
	//	int range_max = (range<<1) + 1;	//2*range+1
	//	int distrSize = searchCandidate * range_max;
	//	cudaMalloc((void**) &d_distribution, sizeof(float) * distrSize);
	//}
}

void free_noiseDst_CUDA(){
	//cudaFree(d_distribution);
}

__global__ void CNMD_DC_kernel(int range_max, float *distribution_prob, float *alphaCeff, int *sideCeff){
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	float *prob = &(distribution_prob[bid * range_max]);
	float alpha = alphaCeff[bid];
	int sideInfo = sideCeff[bid];
	for(int i=tid; i<range_max; i+=THREADNUM1)
		prob[i] = __expf(-alpha * fabsf(i - sideInfo));	//use lapalcian distribution
}

__global__ void CNMD_AC_kernel(int range_max, int range, float *distribution_prob, float *alphaCeff, /*int probLen, float prob, */int *sideCeff, SMF *smf){
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	SMF blkSMF = smf[bid];//
	int probLen = blkSMF.probLen;//
	float *prob = &(distribution_prob[bid * range_max]);
	float alpha = alphaCeff[bid];
	
	//dynamic shared memory
	int *s_sideInfo = (int*)shared;
	float *s_smfProb = (float*)&(s_sideInfo[probLen]);
	if(tid < probLen){
		s_sideInfo[tid] = sideCeff[bid * ML_RANGE + tid];
		s_smfProb[tid] = blkSMF.prob[tid];
		//s_smfProb[tid] = prob[tid];
	}
	__syncthreads();

	//use sum of lapalcian distribution	
	for(int i=tid; i<range_max; i+=THREADNUM1){
		float sum = 0.0;
		int possibleCoeff = i - range;
		for(int j=0; j<probLen; j++)
			sum += s_smfProb[j] * __expf(-alpha * fabsf(possibleCoeff - s_sideInfo[j]));
		prob[i] = sum;
	}
}

void noiseDistribution_CUDA(int range, int indexInBlk, float *alpha, SMF *smf, Trans *overcomplete_trans, SIR *sir){
	int searchRangeWidth = codingOption.FrameWidth - 4 + 1; //used for overcomplete_trans

	if(indexInBlk == 0){
		//DC
		for(int i=0; i<sir->searchCandidate; i++){
			SearchInfo searchInfo = sir->searchInfo[i];
			int top = searchInfo.top;
			int left = searchInfo.left;
			alphaCeff[i] = alpha[top*codingOption.FrameWidth + left];	//laplacian parameter of this coefficient
			sideCeff_DC[i] = overcomplete_trans[top*searchRangeWidth + left].block44[0];	//transform domain
		}

		cudaMemcpy(d_alphaCeff, alphaCeff, sizeof(float) * sir->searchCandidate, cudaMemcpyHostToDevice);
		cudaMemcpy(d_sideCeff_DC, sideCeff_DC, sizeof(int) * sir->searchCandidate, cudaMemcpyHostToDevice);
		
		
		dim3 Grids(sir->searchCandidate,1,1);
		dim3 Blocks(THREADNUM1,1,1);
		CNMD_DC_kernel<<<Grids, Blocks>>>(range, d_distribution, d_alphaCeff, d_sideCeff_DC);
		//printf("distrSize: %d, time: %lfms\n",distrSize,(double)(end.QuadPart-st.QuadPart)*1000.0/CPUFreq);
		//cudaMemcpy(distribution, d_distribution, sizeof(float) * distrSize, cudaMemcpyDeviceToHost);
	}
	else{
		//AC
		int range_max = (range<<1) + 1;	//2*range+1
		int indexInBlkRow = indexInBlk >> 2; //indexInblk / 4
		int indexInBlkCol = indexInBlk & 3;	//indexInblk % 4
		int sideIdx = 0;
		for(int i=0; i<sir->searchCandidate; i++){
			SearchInfo searchInfo = sir->searchInfo[i];
			int top = searchInfo.top;
			int left = searchInfo.left;
			
			alphaCeff[i] = alpha[(top + indexInBlkRow) * codingOption.FrameWidth + left + indexInBlkCol];	//laplacian parameter of this coefficient
			SMF blkSMF = smf[i];	//statistic motion fields
			int *side = &(sideCeff_AC[sideIdx]);
			sideIdx += ML_RANGE;
			int smfIdx = 0;
			for(int m=blkSMF.searchTop; m<=blkSMF.searchBottom; m++){
				int row = m * searchRangeWidth;
				for(int n=blkSMF.searchLeft; n<=blkSMF.searchRight; n++){
					side[smfIdx++] = overcomplete_trans[row + n].block44[indexInBlk];	//transform domain
				}
			}
		}

		cudaMemcpy(d_alphaCeff, alphaCeff, sizeof(float) * sir->searchCandidate, cudaMemcpyHostToDevice);
		cudaMemcpy(d_sideCeff_AC, sideCeff_AC, sizeof(int) * sir->searchCandidate * ML_RANGE, cudaMemcpyHostToDevice);
		cudaMemcpy(d_smf, smf, sizeof(SMF) * sir->searchCandidate, cudaMemcpyHostToDevice);

		dim3 Grids(sir->searchCandidate,1,1);
		dim3 Blocks(THREADNUM1,1,1);
		CNMD_AC_kernel<<<Grids, Blocks, sizeof(int)*ML_RANGE*2>>>(range_max, range, d_distribution, d_alphaCeff, d_sideCeff_AC, d_smf);
		//printf("candidate: %d, range_max: %d, distrSize: %d ,time: %lfms\n", sir->searchCandidate,range_max,distrSize, (double)(end.QuadPart-st.QuadPart)*1000.0/CPUFreq); 
		//system("pause");

		//cudaMemcpy(distribution, d_distribution, sizeof(float) * distrSize, cudaMemcpyDeviceToHost);
	}
	#ifdef SHOW_TIME_INFO
		cudaThreadSynchronize();
	#endif
}


__global__ void CNMP_DC_kernel(float *d_entropy, float *d_LR_intrinsic, int bitOfCoeff, int stepSize, int *d_reconsTransFrame, int range, float *d_distribution, SearchInfo *d_searchInfo){
	int tid = threadIdx.x, bid = blockIdx.x;
	int integralQRange = (1 << (bitOfCoeff-1) ) - 1; //2^(bitOfCoeff-1) - 1
	int reconsCeff = d_reconsTransFrame[bid];	//quantization domain

	//range of quantization index
	int q_if_bit0_lower = reconsCeff << bitOfCoeff;
	int q_if_bit1_lower = ( (reconsCeff << 1) + 1 ) << (bitOfCoeff - 1);
	int q_if_bit0_upper = q_if_bit0_lower + integralQRange;
	int q_if_bit1_upper = q_if_bit1_lower + integralQRange;
	//convert range of quantization index to range of non-quantization value
	int val_if_bit0_lower = q_if_bit0_lower * stepSize;
	int val_if_bit1_lower = q_if_bit1_lower * stepSize;
	int val_if_bit0_upper = (q_if_bit0_upper + 1) * stepSize;
	int val_if_bit1_upper = (q_if_bit1_upper + 1) * stepSize;

	__shared__ float probBuff0[THREADNUM2];
	__shared__ float probBuff1[THREADNUM2];
	float *prob = &(d_distribution[bid * range]);	//noise distribution
	probBuff0[tid] = 0;
	probBuff1[tid] = 0;
	__syncthreads();
	//calculate probability of bit=0
	for(int i=val_if_bit0_lower + tid; i<val_if_bit0_upper; i+=THREADNUM2){
		probBuff0[tid] += prob[i];
	}
	//calculate probability of bit=1
	for(int i=val_if_bit1_lower + tid; i<val_if_bit1_upper; i+=THREADNUM2){
		probBuff1[tid] += prob[i];		
	}

	int candidate = THREADNUM2;
	while(candidate > 1){
		__syncthreads();
		candidate >>= 1;
		if(tid < candidate){
			probBuff0[tid] += probBuff0[tid + candidate];
			probBuff1[tid] += probBuff1[tid + candidate];
		}
	}

	if(tid == 0){		
		float p0 = probBuff0[0] / (probBuff0[0] + probBuff1[0]);		
		if(p0 <= 0.0001f){
			d_LR_intrinsic[d_searchInfo[bid].blkIdx] =  (0.0001f / 0.9999f);
			//atomicAdd(d_entropy, -__log2f(0.0001f)*0.0001f - __log2f(0.9999f)*(0.9999f) );
		}
		else if(p0 >= 0.9999f){
			d_LR_intrinsic[d_searchInfo[bid].blkIdx] =  (0.9999f / 0.0001f);
			//atomicAdd(d_entropy, -__log2f(0.0001f)*0.0001f - __log2f(0.9999f)*(0.9999f) );
		}
		else{
			d_LR_intrinsic[d_searchInfo[bid].blkIdx] = p0 / (1.0f - p0);
			#ifdef FERMI
				atomicAdd(d_entropy, -__log2f(p0)*p0 - __log2f(1.0f-p0)*(1.0f-p0) );		
			#endif
		}
	}
}

__global__ void CNMP_AC_kernel(float *d_entropy, float *d_LR_intrinsic,int idx, int bitOfCoeff, int stepSize, int levelShift, int *d_reconsTransFrame, int range, float *d_distribution, SearchInfo *d_searchInfo){
	int tid = threadIdx.x, bid = blockIdx.x;
	int integralQRange = (1 << (bitOfCoeff-1) ) - 1; //2^(bitOfCoeff-1) - 1
	int reconsCeff = d_reconsTransFrame[bid];	//quantization domain
	
	//range of quantization index			
	int q_if_bit0_lower = (reconsCeff << bitOfCoeff) - levelShift;	//level shift for AC quantization index
	int q_if_bit1_lower = ( ( (reconsCeff << 1) + 1 ) << (bitOfCoeff - 1) ) - levelShift;	//level shift for AC quantization index
	int q_if_bit0_upper = q_if_bit0_lower + integralQRange;
	int q_if_bit1_upper = q_if_bit1_lower + integralQRange;
	if(q_if_bit1_upper > levelShift)
		q_if_bit1_upper = levelShift;
	//convert range of quantization index to range of non-quantization value
	int val_if_bit0_lower, val_if_bit1_lower, val_if_bit0_upper, val_if_bit1_upper;
	if(q_if_bit0_lower <= 0)
		val_if_bit0_lower = (q_if_bit0_lower - 1) * stepSize + 1;
	else
		val_if_bit0_lower = q_if_bit0_lower * stepSize;

	if(q_if_bit1_lower <= 0)
		val_if_bit1_lower = (q_if_bit1_lower - 1) * stepSize + 1;
	else
		val_if_bit1_lower = q_if_bit1_lower * stepSize;
	
	if(q_if_bit0_upper < 0)
		val_if_bit0_upper = q_if_bit0_upper * stepSize + 1;
	else
		val_if_bit0_upper = (q_if_bit0_upper + 1) * stepSize;

	if(q_if_bit1_upper < 0)
		val_if_bit1_upper = q_if_bit1_upper * stepSize + 1;
	else
		val_if_bit1_upper = (q_if_bit1_upper + 1) * stepSize;

	__shared__ float probBuff0[THREADNUM2];
	__shared__ float probBuff1[THREADNUM2];
	val_if_bit0_upper += range;
	val_if_bit1_upper += range;
	int range_max = (range<<1) + 1;
	float *prob = &(d_distribution[bid * range_max]);	//noise distribution
	probBuff0[tid] = 0;
	probBuff1[tid] = 0;
	__syncthreads();
	//calculate probability of bit=0
	for(int i=val_if_bit0_lower + range + tid; i<val_if_bit0_upper; i+=THREADNUM2){
		probBuff0[tid] += prob[i];
	}
	//calculate probability of bit=1
	for(int i=val_if_bit1_lower + range + tid; i<val_if_bit1_upper; i+=THREADNUM2){
		probBuff1[tid] += prob[i];		
	}

	int candidate = THREADNUM2;
	while(candidate > 1){
		__syncthreads();
		candidate >>= 1;
		if(tid < candidate){
			probBuff0[tid] += probBuff0[tid + candidate];
			probBuff1[tid] += probBuff1[tid + candidate];
		}
	}

	if(tid == 0){		
		float p0 = probBuff0[0] / (probBuff0[0] + probBuff1[0]);		
		if(p0 <= 0.0001f){
			d_LR_intrinsic[d_searchInfo[bid].blkIdx] =  (0.0001f / 0.9999f);
			//atomicAdd(d_entropy, -__log2f(0.0001f)*0.0001f - __log2f(0.9999f)*(0.9999f) );
		}
		else if(p0 >= 0.9999f){
			d_LR_intrinsic[d_searchInfo[bid].blkIdx] =  (0.9999f / 0.0001f);
			//atomicAdd(d_entropy, -__log2f(0.0001f)*0.0001f - __log2f(0.9999f)*(0.9999f) );
		}
		else{
			d_LR_intrinsic[d_searchInfo[bid].blkIdx] = p0 / (1.0f - p0);
			#ifdef FERMI
				atomicAdd(d_entropy, -__log2f(p0)*p0 - __log2f(1.0f-p0)*(1.0f-p0) );		
			#endif
		}
	}
}

void condBitProb_CUDA(float *LR_intrinsic, int indexInBlk, int bitOfCoeff, int stepSize, int *reconsTransFrame, int range, SIR *sir){
	dim3 Grids(sir->searchCandidate,1,1);
	dim3 Blocks(THREADNUM2,1,1);
	#ifdef FERMI
		cudaMemset(d_entropy, 0, sizeof(float));
	#endif
	if(indexInBlk == 0){
		//DC
		for(int i=0; i<sir->searchCandidate; i++){
			//conditional bit probability computation
			SearchInfo searchInfo = sir->searchInfo[i];
			reconsCoeff[i] = reconsTransFrame[searchInfo.top * codingOption.FrameWidth + searchInfo.left];	//quantization domain
		}
		cudaMemcpy(d_reconsCoeff, reconsCoeff, sizeof(int) * sir->searchCandidate, cudaMemcpyHostToDevice);
		cudaMemcpy(d_LR_intrinsic, LR_intrinsic, sizeof(float) * Block44Num, cudaMemcpyHostToDevice);

		CNMP_DC_kernel<<<Grids, Blocks>>>(d_entropy, d_LR_intrinsic, bitOfCoeff, stepSize, d_reconsCoeff, range, d_distribution, d_searchInfo);

		cudaMemcpy(LR_intrinsic, d_LR_intrinsic, sizeof(float) * Block44Num, cudaMemcpyDeviceToHost);
	}
	else{
		//AC
		int indexInBlkRow = indexInBlk >> 2; //indexInblk / 4
		int indexInBlkCol = indexInBlk & 3;	//indexInblk % 4
		for(int i=0; i<sir->searchCandidate; i++){
			//conditional bit probability computation
			SearchInfo searchInfo = sir->searchInfo[i];
			int idxInFrame = (searchInfo.top + indexInBlkRow) * codingOption.FrameWidth + searchInfo.left + indexInBlkCol;			
			reconsCoeff[i] = reconsTransFrame[idxInFrame];	//quantization domain			
		}

		cudaMemcpy(d_reconsCoeff, reconsCoeff, sizeof(int) * sir->searchCandidate, cudaMemcpyHostToDevice);
		cudaMemcpy(d_LR_intrinsic, LR_intrinsic, sizeof(float) * Block44Num, cudaMemcpyHostToDevice);

		CNMP_AC_kernel<<<Grids, Blocks>>>(d_entropy, d_LR_intrinsic, indexInBlk,bitOfCoeff, stepSize, ACLevelShift[indexInBlk], d_reconsCoeff, range, d_distribution, d_searchInfo);

		cudaMemcpy(LR_intrinsic, d_LR_intrinsic, sizeof(float) * Block44Num, cudaMemcpyDeviceToHost);	
	}

}
#include <stdlib.h>
#include <math.h>
#include "noiseModel.h"
#include "reconstruction.h"
#include "omp.h"

#include <stdio.h>

void noiseParameter(int *transFrame, float *alpha, unsigned char *BlockModeFlag){
	
	//calculate mean and variance of each coefficient band
	float bandMean[BLOCKSIZE44] = {0};
	float bandVar[BLOCKSIZE44] = {0};
	for(int i=0; i<Block44Height; i++){
		int top = i << 2;
		for(int j=0; j<Block44Width; j++){
			//for each 4x4 block
			int left = j << 2;

			//zig-zag order
			int band = 0;	//from DC band
			int index = zigzag[band];	//index in block (zig-zag order)
			while(QtableBits[index] > 0){
				//for each band
				int coeffIdx = (top+(index>>2)) * (codingOption.FrameWidth) + left+ (index&3);	//index>>2 -> index/4, index&3 -> index%4
				float coefficient = abs((float)transFrame[coeffIdx]);
				bandMean[band] += coefficient;	//|T|
				bandVar[band] += coefficient * coefficient;	//T^2
				transFrame[coeffIdx] = (int)coefficient;
				band++;
				index = zigzag[band];
			}
		}
	}

	//zig-zag order
	int band = 0;	//from DC band
	int index = zigzag[band];	//index in block (zig-zag order)
	while(QtableBits[index] > 0){
		//for each band
		bandMean[band] /= (float)Block44Num;	//E[|T|]
		bandVar[band] = bandVar[band] / (float)Block44Num - bandMean[band] * bandMean[band];	//E[|T|^2] - E[|T|]^2
		band++;
		index = zigzag[band];
	}
	
	//correlation noise model parameter computation
	for(int i=0; i<Block44Height; i++){
		int top = i << 2;
		int blk88row = (i>>1) * Block88Width;
		for(int j=0; j<Block44Width; j++){
			//for each 4x4 block
			if(BlockModeFlag[blk88row + (j>>1)] == WZ_B){
				int left = j << 2;

				//zig-zag order
				int band = 0;	//from DC band
				int index = zigzag[band];	//index in block (zig-zag order)
				while(QtableBits[index] > 0){
					//for each band
					int coeffIdx = (top+(index>>2)) * (codingOption.FrameWidth) + left+ (index&3);	//index>>2 -> index/4, index&3 -> index%4
					float dst = (float)transFrame[coeffIdx] - bandMean[band];
					float dst2 = dst * dst;
					if(dst2 > bandVar[band])
						alpha[coeffIdx] = sqrt(2 / dst2);
					else
						alpha[coeffIdx] = sqrt(2 / bandVar[band]);
					band++;
					index = zigzag[band];
				}
			}
		}
	}
	
}

float *noiseDistribution(int range, int indexInBlk, float *alpha, SMF *smf, Trans *overcomplete_trans, SIR *sir){
	float *distribution;
	int distrIdx = 0;
	int searchRangeWidth = codingOption.FrameWidth - 4 + 1; //used for overcomplete_trans
	if(indexInBlk == 0){
		//DC		
		distribution = (float*)malloc(sizeof(float) * sir->searchCandidate * range);
		for(int i=0; i<sir->searchCandidate; i++){
			SearchInfo searchInfo = sir->searchInfo[i];
			int top = searchInfo.top;
			int left = searchInfo.left;
			float *prob = &(distribution[distrIdx]);
			distrIdx += range;
			float alphaCeff = alpha[top*codingOption.FrameWidth + left];	//laplacian parameter of this coefficient
			int sideCeff = overcomplete_trans[top*searchRangeWidth + left].block44[0];	//transform domain			
			for(int k=0; k<range; k++)
				prob[k] = exp(-alphaCeff * (float)abs(k - sideCeff));	//use lapalcian distribution
		}
	}
	else{
		//AC
		int range_max = (range<<1) + 1;	//2*range+1
		distribution = (float*)malloc(sizeof(float) * sir->searchCandidate * range_max);
		int indexInBlkRow = indexInBlk >> 2; //indexInblk / 4
		int indexInBlkCol = indexInBlk & 3;	//indexInblk % 4
		for(int i=0; i<sir->searchCandidate; i++){
			SearchInfo searchInfo = sir->searchInfo[i];
			int top = searchInfo.top;
			int left = searchInfo.left;
			
			float alphaCeff = alpha[(top + indexInBlkRow) * codingOption.FrameWidth + left + indexInBlkCol];	//laplacian parameter of this coefficient
			SMF blkSMF = smf[i];	//statistic motion fields
			int sideCeff[ML_RANGE];
			int smfIdx = 0;
			for(int m=blkSMF.searchTop; m<=blkSMF.searchBottom; m++){
				int row = m * searchRangeWidth;
				for(int n=blkSMF.searchLeft; n<=blkSMF.searchRight; n++){
					sideCeff[smfIdx++] = overcomplete_trans[row + n].block44[indexInBlk];	//transform domain
				}
			}

			//use sum of lapalcian distribution
			float sum;
			float *prob = &(distribution[distrIdx]);
			distrIdx += range_max;
			for(int k=0; k<range_max; k++){
				sum = 0.0;
				int possibleCoeff = k - range;
				for(int m=0; m<smfIdx; m++)
					sum += blkSMF.prob[m] * exp(-alphaCeff * (float)abs(possibleCoeff - sideCeff[m]));
				prob[k] = sum;
			}		
		}
	}
	return distribution;
}

float *noiseDistribution_OpenMP(int range, int indexInBlk, float *alpha, SMF *smf, Trans *overcomplete_trans, SIR *sir){
	float *distribution;
	int searchRangeWidth = codingOption.FrameWidth - 4 + 1; //used for overcomplete_trans
	if(indexInBlk == 0){
		//DC		
		distribution = (float*)malloc(sizeof(float) * sir->searchCandidate * range);
#pragma omp parallel for
		for(int i=0; i<sir->searchCandidate; i++){
			SearchInfo searchInfo = sir->searchInfo[i];
			int top = searchInfo.top;
			int left = searchInfo.left;
			float *prob = &(distribution[i * range]);
			float alphaCeff = alpha[top*codingOption.FrameWidth + left];	//laplacian parameter of this coefficient
			int sideCeff = overcomplete_trans[top*searchRangeWidth + left].block44[0];	//transform domain			
			for(int k=0; k<range; k++)
				prob[k] = exp(-alphaCeff * (float)abs(k - sideCeff));	//use lapalcian distribution
		}
	}
	else{
		//AC
		int range_max = (range<<1) + 1;	//2*range+1
		distribution = (float*)malloc(sizeof(float) * sir->searchCandidate * range_max);
		int indexInBlkRow = indexInBlk >> 2; //indexInblk / 4
		int indexInBlkCol = indexInBlk & 3;	//indexInblk % 4
#pragma omp parallel for
		for(int i=0; i<sir->searchCandidate; i++){
			SearchInfo searchInfo = sir->searchInfo[i];
			int top = searchInfo.top;
			int left = searchInfo.left;
			
			float alphaCeff = alpha[(top + indexInBlkRow) * codingOption.FrameWidth + left + indexInBlkCol];	//laplacian parameter of this coefficient
			SMF blkSMF = smf[i];	//statistic motion fields
			int sideCeff[ML_RANGE];
			int smfIdx = 0;
			for(int m=blkSMF.searchTop; m<=blkSMF.searchBottom; m++){
				int row = m * searchRangeWidth;
				for(int n=blkSMF.searchLeft; n<=blkSMF.searchRight; n++){
					sideCeff[smfIdx++] = overcomplete_trans[row + n].block44[indexInBlk];	//transform domain
				}
			}

			//use sum of lapalcian distribution
			float sum;
			float *prob = &(distribution[i * range_max]);
			for(int k=0; k<range_max; k++){
				sum = 0.0;
				int possibleCoeff = k - range;
				for(int m=0; m<smfIdx; m++)
					sum += blkSMF.prob[m] * exp(-alphaCeff * (float)abs(possibleCoeff - sideCeff[m]));
				prob[k] = sum;
			}
		}
	}
	return distribution;
}


void condBitProb(float *LLR_intrinsic, int indexInBlk, int bitOfCoeff, int stepSize, int *reconsTransFrame, int range, float *distribution, SIR *sir){
	int integralQRange = (1 << (bitOfCoeff-1) ) - 1; //2^(bitOfCoeff-1) - 1

	if(indexInBlk == 0){
		//DC
		for(int i=0; i<sir->searchCandidate; i++){
			//conditional bit probability computation
			SearchInfo searchInfo = sir->searchInfo[i];
			int reconsCeff = reconsTransFrame[searchInfo.top * codingOption.FrameWidth + searchInfo.left];	//quantization domain
			//int blk44Idx = (searchInfo.top>>2) * Block44Width + (searchInfo.left>>2);
			float *prob = &(distribution[i * range]);	//noise distribution

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
			
			//calculate probability of bit=0
			float p0 = 0.0, p1 = 0.0;
			for(int k=val_if_bit0_lower; k<val_if_bit0_upper; k++){
				//p0 += exp(-alphaCeff * (double)abs(k - sideCeff)) / 2.0;	//use lapalcian distribution
				p0 += prob[k];
			}
			//calculate probability of bit=1
			for(int k=val_if_bit1_lower; k<val_if_bit1_upper; k++){
				//p1 += exp(-alphaCeff * (double)abs(k - sideCeff)) / 2.0;	//use lapalcian distribution
				p1 += prob[k];
			}
			//p0 = p0 * alphaCeff / 2.0;
			//p1 = p1 * alphaCeff / 2.0;

			p0 = p0 / (p0 + p1);
			p1 = 1 - p0;
			if(p0 <= 0.0001)
				LLR_intrinsic[searchInfo.blkIdx] =  (float)(0.0001 / 0.9999);
			else if(p0 >= 0.9999)
				LLR_intrinsic[searchInfo.blkIdx] =  (float)(0.9999 / 0.0001);
			else
				LLR_intrinsic[searchInfo.blkIdx] = p0 / p1;
		}
	}
	else{
		//AC
		int indexInBlkRow = indexInBlk >> 2; //indexInblk / 4
		int indexInBlkCol = indexInBlk & 3;	//indexInblk % 4
		int levelShift = ACLevelShift[indexInBlk];
		int range_max = (range<<1) + 1;
		for(int i=0; i<sir->searchCandidate; i++){
			//conditional bit probability computation
			SearchInfo searchInfo = sir->searchInfo[i];
			int idxInFrame = (searchInfo.top + indexInBlkRow) * codingOption.FrameWidth + searchInfo.left + indexInBlkCol;
			int reconsCeff = reconsTransFrame[idxInFrame];	//quantization domain
			//int blk44Idx = (searchInfo.top>>2) * Block44Width + (searchInfo.left>>2);
			float *prob = &(distribution[i * range_max]);	//noise distribution

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

			//calculate probability of bit=0
			float p0 = 0.0, p1 = 0.0;
			val_if_bit0_upper += range;
			for(int k=val_if_bit0_lower+range; k<val_if_bit0_upper; k++){
				//p0 += alphaCeff * exp(-alphaCeff * (double)abs(k - sideCeff)) / 2.0;	//use lapalcian distribution
				//p0 += exp(-alphaCeff * (double)abs(k - sideCeff));	//use lapalcian distribution
				p0 += prob[k];
			}
			//calculate probability of bit=1
			val_if_bit1_upper += range;
			for(int k=val_if_bit1_lower+range; k<val_if_bit1_upper; k++){
				//p1 += alphaCeff * exp(-alphaCeff * (double)abs(k - sideCeff)) / 2.0;	//use lapalcian distribution
				//p1 += exp(-alphaCeff * (double)abs(k - sideCeff));	//use lapalcian distribution
				p1 += prob[k];
			}
			//p0 = p0 * alphaCeff / 2.0;
			//p1 = p1 * alphaCeff / 2.0;
			
			p0 = p0 / (p0 + p1);
			p1 = 1 - p0;
			if(p0 <= 0.0001)
				LLR_intrinsic[searchInfo.blkIdx] =  (float)(0.0001 / 0.9999);
			else if(p0 >= 0.9999)
				LLR_intrinsic[searchInfo.blkIdx] =  (float)(0.9999 / 0.0001);
			else
				LLR_intrinsic[searchInfo.blkIdx] = p0 / p1;
		}		
	}

}



void condBitProb_OpenMP(float *LLR_intrinsic, int indexInBlk, int bitOfCoeff, int stepSize, int *reconsTransFrame, int range, float *distribution, SIR *sir){
	
	int integralQRange = (1 << (bitOfCoeff-1) ) - 1; //2^(bitOfCoeff-1) - 1

	if(indexInBlk == 0){
		//DC
#pragma omp parallel for
		for(int i=0; i<sir->searchCandidate; i++){
			//conditional bit probability computation
			SearchInfo searchInfo = sir->searchInfo[i];
			int reconsCeff = reconsTransFrame[searchInfo.top * codingOption.FrameWidth + searchInfo.left];	//quantization domain
			//int blk44Idx = (searchInfo.top>>2) * Block44Width + (searchInfo.left>>2);
			float *prob = &(distribution[i * range]);	//noise distribution

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
			
			//calculate probability of bit=0
			float p0 = 0.0, p1 = 0.0;
			for(int k=val_if_bit0_lower; k<val_if_bit0_upper; k++){
				//p0 += exp(-alphaCeff * (double)abs(k - sideCeff)) / 2.0;	//use lapalcian distribution
				p0 += prob[k];
			}
			//calculate probability of bit=1
			for(int k=val_if_bit1_lower; k<val_if_bit1_upper; k++){
				//p1 += exp(-alphaCeff * (double)abs(k - sideCeff)) / 2.0;	//use lapalcian distribution
				p1 += prob[k];
			}
			//p0 = p0 * alphaCeff / 2.0;
			//p1 = p1 * alphaCeff / 2.0;

			p0 = p0 / (p0 + p1);
			p1 = 1 - p0;
			if(p0 <= 0.0001)
				LLR_intrinsic[searchInfo.blkIdx] =  (float)(0.0001 / 0.9999);
			else if(p0 >= 0.9999)
				LLR_intrinsic[searchInfo.blkIdx] =  (float)(0.9999 / 0.0001);
			else
				LLR_intrinsic[searchInfo.blkIdx] = p0 / p1;
		}
	}
	else{
		//AC
		int indexInBlkRow = indexInBlk >> 2; //indexInblk / 4
		int indexInBlkCol = indexInBlk & 3;	//indexInblk % 4
		int levelShift = ACLevelShift[indexInBlk];
		int range_max = (range<<1) + 1;
#pragma omp parallel for
		for(int i=0; i<sir->searchCandidate; i++){
			//conditional bit probability computation
			SearchInfo searchInfo = sir->searchInfo[i];
			int idxInFrame = (searchInfo.top + indexInBlkRow) * codingOption.FrameWidth + searchInfo.left + indexInBlkCol;
			int reconsCeff = reconsTransFrame[idxInFrame];	//quantization domain
			//int blk44Idx = (searchInfo.top>>2) * Block44Width + (searchInfo.left>>2);
			float *prob = &(distribution[i * range_max]);	//noise distribution

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

			//calculate probability of bit=0
			float p0 = 0.0, p1 = 0.0;
			val_if_bit0_upper += range;
			for(int k=val_if_bit0_lower+range; k<val_if_bit0_upper; k++){
				//p0 += alphaCeff * exp(-alphaCeff * (double)abs(k - sideCeff)) / 2.0;	//use lapalcian distribution
				//p0 += exp(-alphaCeff * (double)abs(k - sideCeff));	//use lapalcian distribution
				p0 += prob[k];
			}
			//calculate probability of bit=1
			val_if_bit1_upper += range;
			for(int k=val_if_bit1_lower+range; k<val_if_bit1_upper; k++){
				//p1 += alphaCeff * exp(-alphaCeff * (double)abs(k - sideCeff)) / 2.0;	//use lapalcian distribution
				//p1 += exp(-alphaCeff * (double)abs(k - sideCeff));	//use lapalcian distribution
				p1 += prob[k];
			}
			//p0 = p0 * alphaCeff / 2.0;
			//p1 = p1 * alphaCeff / 2.0;
			
			p0 = p0 / (p0 + p1);
			p1 = 1 - p0;
			if(p0 <= 0.0001)
				LLR_intrinsic[searchInfo.blkIdx] =  (float)(0.0001 / 0.9999);
			else if(p0 >= 0.9999)
				LLR_intrinsic[searchInfo.blkIdx] =  (float)(0.9999 / 0.0001);
			else
				LLR_intrinsic[searchInfo.blkIdx] = p0 / p1;
		}		
	}

}


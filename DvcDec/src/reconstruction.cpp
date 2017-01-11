#include <math.h>
#include "reconstruction.h"

//use for quantization
int *Qtable;
int *QtableBits;
int QbitsNumPerBlk = 0;	//amount of bitplanes per frame
int ACLevelShift[BLOCKSIZE44] = {0};	//AC level shift (let all coefficients >= 0)
int QuantizationTable[9][BLOCKSIZE44] = {
	{0},
	{ 16,8,0,0,
	  8,0,0,0,
	  0,0,0,0,
	  0,0,0,0  },
	{ 32,8,0,0,
	  8,0,0,0,
	  0,0,0,0,
	  0,0,0,0  },
	{ 32,8,4,0,
	  8,4,0,0,
	  4,0,0,0,
	  0,0,0,0  },
	{ 32,16,8,4,
	  16,8,4,0,
	  8,4,0,0,
	  4,0,0,0  },
    { 32,16,8,4,
	  16,8,4,4,
	  8,4,4,0,
	  4,4,0,0  },
	{ 64,16,8,8,
	  16,8,8,4,
	  8,8,4,4,
	  8,4,4,0  },
	{ 64,32,16,8,
	  32,16,8,4,
	  16,8,4,4,
	  8,4,4,0  },
	{ 128,64,32,16,
	  64,32,16,8,
	  32,16,8,4,
	  16,8,4,0  }
};
int QuantizationTableBits[9][BLOCKSIZE44] = {
	{0},
	{ 4,3,0,0,
	  3,0,0,0,
	  0,0,0,0,
	  0,0,0,0  },
	{ 5,3,0,0,
	  3,0,0,0,
	  0,0,0,0,
	  0,0,0,0  },
	{ 5,3,2,0,
	  3,2,0,0,
	  2,0,0,0,
	  0,0,0,0  },
	{ 5,4,3,2,
	  4,3,2,0,
	  3,2,0,0,
	  2,0,0,0  },
    { 5,4,3,2,
	  4,3,2,2,
	  3,2,2,0,
	  2,2,0,0  },
	{ 6,4,3,3,
	  4,3,3,2,
	  3,3,2,2,
	  3,2,2,0  },
	{ 6,5,4,3,
	  5,4,3,2,
	  4,3,2,2,
	  3,2,2,0  },
	{ 7,6,5,4,
	  6,5,4,3,
	  5,4,3,2,
	  4,3,2,0  }
};


void quantize44(int *blk, int *stepSize){
	//DC
	blk[0] /= stepSize[0];
	//AC
	for(int i=1; i<BLOCKSIZE44; i++){
		if(stepSize[i] > 0)	
			blk[i] = blk[i] / stepSize[i] + ACLevelShift[i];
		else
			blk[i] = 0;
	}
}


/*void reconstruction(int *reconsTransFrame, int *sideTransFrame, int band, double *alpha, int *stepSize, unsigned char *skipBlockFlag){

	for(int i=0; i<Block44Height; i++){
		int top = i << 2;
		int blk88row = (i>>1) * Block88Width;
		for(int j=0; j<Block44Width; j++){
			//for each 4x4 block
			int left = j << 2;

			if(skipBlockFlag[blk88row + (j>>1)] == 0){
				//zig-zag order
				//DC band reconstruction
				int coeffIdx = top * codingOption.FrameWidth + left;
				int stepsize = stepSize[0];
				int qIdx = reconsTransFrame[coeffIdx];
				int val_lower = qIdx * stepsize;
				int val_upper = (qIdx + 1) * stepsize;
				int sideCeff = sideTransFrame[coeffIdx];
				double alphaCeff = alpha[coeffIdx];
				
				if(alphaCeff <= 0.0001){
					sideTransFrame[coeffIdx] = (val_lower + val_upper) / 2;
				}
				else{
					if(sideCeff < val_lower){
						sideTransFrame[coeffIdx] = val_lower + ROUND( 1.0 / alphaCeff + (double)stepsize / (1.0 - exp(alphaCeff * (double)stepsize)) );
					}
					else if(sideCeff >= val_upper){
						sideTransFrame[coeffIdx] = val_upper - ROUND( 1.0 / alphaCeff + (double)stepsize / (1.0 - exp(alphaCeff * (double)stepsize)) );
					}
					else{
						double r = sideCeff - val_lower;
						double delta = val_upper - sideCeff;
						double exp_alpha_r = exp(-alphaCeff * r);
						double exp_alpha_delta = exp(-alphaCeff * delta);
						sideTransFrame[coeffIdx] = sideCeff + ROUND( 
							( (r + 1.0 / alphaCeff) * exp_alpha_r - (delta + 1.0 / alphaCeff) * exp_alpha_delta ) / ( 2.0 - (exp_alpha_r + exp_alpha_delta) )
							);
					}
				}
				
				//AC band reconstruction
				for(int k=1; k<=band; k++){
					int index = zigzag[k];	//index in block (zig-zag order)
					//for each band
					coeffIdx = (top+(index>>2)) * (codingOption.FrameWidth) + left+ (index&3);	//index>>2 -> index/4, index&3 -> index%4
					stepsize = stepSize[index];
					qIdx = reconsTransFrame[coeffIdx] - ACLevelShift[index];
					if(qIdx == 0){
						val_lower = -stepsize + 1;
						val_upper = stepsize;
					}
					else if(qIdx > 0){
						val_lower = qIdx * stepsize;
						val_upper = (qIdx + 1) * stepsize;
					}
					else{
						val_lower = (qIdx - 1) * stepsize + 1;
						val_upper = qIdx * stepsize + 1;
					}

					sideCeff = sideTransFrame[coeffIdx];
					alphaCeff = alpha[coeffIdx];
					if(alphaCeff <= 0.0001){
						sideTransFrame[coeffIdx] = (val_lower + val_upper) / 2;
					}
					else{
						if(sideCeff < val_lower){
							sideTransFrame[coeffIdx] = val_lower + ROUND( 1.0 / alphaCeff + (double)stepsize / (1.0 - exp(alphaCeff * (double)stepsize)) );
						}
						else if(sideCeff >= val_upper){
							sideTransFrame[coeffIdx] = val_upper - ROUND( 1.0 / alphaCeff + (double)stepsize / (1.0 - exp(alphaCeff * (double)stepsize)) );
						}
						else{
							double r = sideCeff - val_lower;
							double delta = val_upper - sideCeff;
							double exp_alpha_r = exp(-alphaCeff * r);
							double exp_alpha_delta = exp(-alphaCeff * delta);
							sideTransFrame[coeffIdx] = sideCeff + ROUND( 
								( (r + 1.0 / alphaCeff) * exp_alpha_r - (delta + 1.0 / alphaCeff) * exp_alpha_delta ) / ( 2.0 - (exp_alpha_r + exp_alpha_delta) )
								);
						}
					}
				}			
			}

		}
	}
}*/

void reconstruction(int *reconsTransFrame, int *sideTransFrame, int indexInblk, float *alpha, int *stepSize, unsigned char *BlockModeFlag, unsigned char *deblockFlag){
	int blkRowIdx = indexInblk >> 2;
	int blkColIdx = indexInblk & 3;
	int stepsize = stepSize[indexInblk];
	if(indexInblk == 0){
		//DC band reconstruction
		for(int i=0; i<Block44Height; i++){
			int top = i << 2;
			int blk88row = (i>>1) * Block88Width;
			int blkRow = top * codingOption.FrameWidth;
			for(int j=0; j<Block44Width; j++){
				//for each 4x4 block

				if(BlockModeFlag[blk88row + (j>>1)] == WZ_B){
					//DC band reconstruction
					int coeffIdx = blkRow + (j<<2);
					int qIdx = reconsTransFrame[coeffIdx];
					int val_lower = qIdx * stepsize;
					int val_upper = (qIdx + 1) * stepsize;
					int sideCeff = sideTransFrame[coeffIdx];
					//int sideCeff = sideTransML[coeffIdx];
					double alphaCeff = alpha[coeffIdx];
					if(alphaCeff <= 0.0001){
							sideTransFrame[coeffIdx] = (val_lower + val_upper) / 2;
					}
					else{
						if(sideCeff < val_lower){
							sideTransFrame[coeffIdx] = val_lower + ROUND( 1.0 / alphaCeff + (double)stepsize / (1.0 - exp(alphaCeff * (double)stepsize)) );
						}
						else if(sideCeff >= val_upper){
							sideTransFrame[coeffIdx] = val_upper - ROUND( 1.0 / alphaCeff + (double)stepsize / (1.0 - exp(alphaCeff * (double)stepsize)) );
						}
						else{
							double r = sideCeff - val_lower;
							double delta = val_upper - sideCeff;
							double exp_alpha_r = exp(-alphaCeff * r);
							double exp_alpha_delta = exp(-alphaCeff * delta);
							sideTransFrame[coeffIdx] = sideCeff + ROUND( 
								( (r + 1.0 / alphaCeff) * exp_alpha_r - (delta + 1.0 / alphaCeff) * exp_alpha_delta ) / ( 2.0 - (exp_alpha_r + exp_alpha_delta) ) 
								);
						}
					}
					deblockFlag[coeffIdx] = ((sideTransFrame[coeffIdx] ^ sideCeff) != 0) ? 1 : 0;
					//sideTransFrame[coeffIdx] = reconsTransFrame[coeffIdx];
				}
			}
		}
	}
	else{
		//AC band reconstruction
		int levelShift = ACLevelShift[indexInblk];
		for(int i=0; i<Block44Height; i++){
			int top = i << 2;
			int blk88row = (i>>1) * Block88Width;
			int blkRow = (top+blkRowIdx) * codingOption.FrameWidth;
			for(int j=0; j<Block44Width; j++){
				//for each 4x4 block

				if(BlockModeFlag[blk88row + (j>>1)] == WZ_B){				
					int coeffIdx = blkRow + (j<<2) + blkColIdx;	//index>>2 -> index/4, index&3 -> index%4
					int qIdx = reconsTransFrame[coeffIdx] - levelShift;
					int val_lower, val_upper;
					int sideCeff = sideTransFrame[coeffIdx];
					//int sideCeff = sideTransML[coeffIdx];
					double alphaCeff = alpha[coeffIdx];
					if(qIdx == 0){
						val_lower = -stepsize + 1;
						val_upper = stepsize;
					}
					else if(qIdx > 0){
						val_lower = qIdx * stepsize;
						val_upper = (qIdx + 1) * stepsize;
					}
					else{
						val_lower = (qIdx - 1) * stepsize + 1;
						val_upper = qIdx * stepsize + 1;
					}
					
					if(alphaCeff <= 0.0001){
						sideTransFrame[coeffIdx] = (val_lower + val_upper) / 2;
					}
					else{
						if(sideCeff < val_lower){
							sideTransFrame[coeffIdx] = val_lower + ROUND( 1.0 / alphaCeff + (double)stepsize / (1.0 - exp(alphaCeff * (double)stepsize)) );
						}
						else if(sideCeff >= val_upper){
							sideTransFrame[coeffIdx] = val_upper - ROUND( 1.0 / alphaCeff + (double)stepsize / (1.0 - exp(alphaCeff * (double)stepsize)) );
						}
						else{
							double r = sideCeff - val_lower;
							double delta = val_upper - sideCeff;
							double exp_alpha_r = exp(-alphaCeff * r);
							double exp_alpha_delta = exp(-alphaCeff * delta);
							sideTransFrame[coeffIdx] = sideCeff + ROUND( 
								( (r + 1.0 / alphaCeff) * exp_alpha_r - (delta + 1.0 / alphaCeff) * exp_alpha_delta ) / ( 2.0 - (exp_alpha_r + exp_alpha_delta) ) 
								);
						}
					}
					deblockFlag[coeffIdx] = ((sideTransFrame[coeffIdx] ^ sideCeff) != 0) ? 1 : 0;
					//sideTransFrame[coeffIdx] = reconsTransFrame[coeffIdx];
				}

			}
		}
	}
}

void reconstruction_OpenMP(int *reconsTransFrame, int *sideTransFrame, int indexInblk, float *alpha, int *stepSize, unsigned char *BlockModeFlag, unsigned char *deblockFlag){
	int blkRowIdx = indexInblk >> 2;
	int blkColIdx = indexInblk & 3;
	int stepsize = stepSize[indexInblk];
	if(indexInblk == 0){
		//DC band reconstruction
		#pragma omp parallel for
		for(int i=0; i<Block44Height; i++){
			int top = i << 2;
			int blk88row = (i>>1) * Block88Width;
			int blkRow = top * codingOption.FrameWidth;
			for(int j=0; j<Block44Width; j++){
				//for each 4x4 block

				if(BlockModeFlag[blk88row + (j>>1)] == WZ_B){
					//DC band reconstruction
					int coeffIdx = blkRow + (j<<2);
					int qIdx = reconsTransFrame[coeffIdx];
					int val_lower = qIdx * stepsize;
					int val_upper = (qIdx + 1) * stepsize;
					int sideCeff = sideTransFrame[coeffIdx];
					//int sideCeff = sideTransML[coeffIdx];
					double alphaCeff = alpha[coeffIdx];
					if(alphaCeff <= 0.0001){
							sideTransFrame[coeffIdx] = (val_lower + val_upper) / 2;
					}
					else{
						if(sideCeff < val_lower){
							sideTransFrame[coeffIdx] = val_lower + ROUND( 1.0 / alphaCeff + (double)stepsize / (1.0 - exp(alphaCeff * (double)stepsize)) );
						}
						else if(sideCeff >= val_upper){
							sideTransFrame[coeffIdx] = val_upper - ROUND( 1.0 / alphaCeff + (double)stepsize / (1.0 - exp(alphaCeff * (double)stepsize)) );
						}
						else{
							double r = sideCeff - val_lower;
							double delta = val_upper - sideCeff;
							double exp_alpha_r = exp(-alphaCeff * r);
							double exp_alpha_delta = exp(-alphaCeff * delta);
							sideTransFrame[coeffIdx] = sideCeff + ROUND( 
								( (r + 1.0 / alphaCeff) * exp_alpha_r - (delta + 1.0 / alphaCeff) * exp_alpha_delta ) / ( 2.0 - (exp_alpha_r + exp_alpha_delta) ) 
								);
						}
					}
					deblockFlag[coeffIdx] = ((sideTransFrame[coeffIdx] ^ sideCeff) != 0) ? 1 : 0;
					//sideTransFrame[coeffIdx] = reconsTransFrame[coeffIdx];
				}
			}
		}
	}
	else{
		//AC band reconstruction
		int levelShift = ACLevelShift[indexInblk];
		#pragma omp parallel for
		for(int i=0; i<Block44Height; i++){
			int top = i << 2;
			int blk88row = (i>>1) * Block88Width;
			int blkRow = (top+blkRowIdx) * codingOption.FrameWidth;
			for(int j=0; j<Block44Width; j++){
				//for each 4x4 block

				if(BlockModeFlag[blk88row + (j>>1)] == WZ_B){				
					int coeffIdx = blkRow + (j<<2) + blkColIdx;	//index>>2 -> index/4, index&3 -> index%4
					int qIdx = reconsTransFrame[coeffIdx] - levelShift;
					int val_lower, val_upper;
					int sideCeff = sideTransFrame[coeffIdx];
					//int sideCeff = sideTransML[coeffIdx];
					double alphaCeff = alpha[coeffIdx];
					if(qIdx == 0){
						val_lower = -stepsize + 1;
						val_upper = stepsize;
					}
					else if(qIdx > 0){
						val_lower = qIdx * stepsize;
						val_upper = (qIdx + 1) * stepsize;
					}
					else{
						val_lower = (qIdx - 1) * stepsize + 1;
						val_upper = qIdx * stepsize + 1;
					}
					
					if(alphaCeff <= 0.0001){
						sideTransFrame[coeffIdx] = (val_lower + val_upper) / 2;
					}
					else{
						if(sideCeff < val_lower){
							sideTransFrame[coeffIdx] = val_lower + ROUND( 1.0 / alphaCeff + (double)stepsize / (1.0 - exp(alphaCeff * (double)stepsize)) );
						}
						else if(sideCeff >= val_upper){
							sideTransFrame[coeffIdx] = val_upper - ROUND( 1.0 / alphaCeff + (double)stepsize / (1.0 - exp(alphaCeff * (double)stepsize)) );
						}
						else{
							double r = sideCeff - val_lower;
							double delta = val_upper - sideCeff;
							double exp_alpha_r = exp(-alphaCeff * r);
							double exp_alpha_delta = exp(-alphaCeff * delta);
							sideTransFrame[coeffIdx] = sideCeff + ROUND( 
								( (r + 1.0 / alphaCeff) * exp_alpha_r - (delta + 1.0 / alphaCeff) * exp_alpha_delta ) / ( 2.0 - (exp_alpha_r + exp_alpha_delta) ) 
								);
						}
					}
					deblockFlag[coeffIdx] = ((sideTransFrame[coeffIdx] ^ sideCeff) != 0) ? 1 : 0;
					//sideTransFrame[coeffIdx] = reconsTransFrame[coeffIdx];
				}

			}
		}
	}
}
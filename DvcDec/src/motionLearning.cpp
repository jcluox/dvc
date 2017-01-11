#include <stdlib.h>
#include <math.h>
#include "motionLearning.h"
#include "omp.h"

void initSMF(SMF *smf, SIR *sir){
	int searchRangeRight = codingOption.FrameWidth - 4;
	int searchRangeBottom = codingOption.FrameHeight - 4;

	for(int i=0; i<sir->searchCandidate; i++){
		//for each smf
		SearchInfo searchInfo = sir->searchInfo[i];
		int top = searchInfo.top;
		int left = searchInfo.left;

		int searchTop = top - LEARNSEARCHRANGE;
		int searchBottom = top + LEARNSEARCHRANGE;
		int searchLeft = left - LEARNSEARCHRANGE;
		int searchRight = left + LEARNSEARCHRANGE;
		if(searchTop < 0)
			searchTop = 0;
		if(searchBottom > searchRangeBottom)
			searchBottom = searchRangeBottom;			
		if(searchLeft < 0)
			searchLeft = 0;
		if(searchRight > searchRangeRight)
			searchRight = searchRangeRight;

		//smf[i].prob = (double*)malloc(sizeof(double) * (searchRight - searchLeft + 1) * (searchBottom - searchTop + 1));
		int probIdx = 0;
		for(int m=searchTop; m<=searchBottom; m++)
			for(int n=searchLeft; n<=searchRight; n++){
				if(m==top && n==left){
					smf[i].blkIdx = probIdx;	
					smf[i].prob[probIdx++] = 1.0;
				}
				else
					smf[i].prob[probIdx++] = 0.0;
			}
		smf[i].searchTop = searchTop;
		smf[i].searchBottom = searchBottom;
		smf[i].searchLeft = searchLeft;
		smf[i].searchRight = searchRight;
		smf[i].searchWidth = searchRight - searchLeft + 1;
		smf[i].probLen = probIdx;
	}
}

void init_si_ML(int *si_ML, Trans *transFrame){
	int searchRangeWidth = codingOption.FrameWidth - 4 + 1;
	for(int i=0; i<Block44Height; i++){
		int top = i<<2;
		int row = top * searchRangeWidth;
		for(int j=0; j<Block44Width; j++){
			//for each block
			int left = j<<2;
			int idx = row + left;

			for(int m=0; m<4; m++){
				int blkRow = m<<2;
				int frameRow = (top + m) * codingOption.FrameWidth + left;
				for(int n=0; n<4; n++){
					si_ML[frameRow + n] = transFrame[idx].block44[blkRow + n];
				}
			}
		}
	}
}

void updateSMF(SMF *smf, unsigned char *reconsFrame, unsigned char *sideInfoFrame, SIR *sir){

	for(int i=0; i<sir->searchCandidate; i++){
		//for each smf
		SearchInfo searchInfo = sir->searchInfo[i];
		int top = searchInfo.top;
		int left = searchInfo.left;

		float  block[BLOCKSIZE44];
		int blkIdx = 0;
		int rowInFrame = top * codingOption.FrameWidth + left;
		for(int m=0; m<4; m++){
			//int blkRow = m << 2;
			//int rowInFrame = (top+m) * codingOption.FrameWidth + left;
			//for(int n=0; n<4; n++)
			block[blkIdx++] = (float)reconsFrame[rowInFrame];
			block[blkIdx++] = (float)reconsFrame[rowInFrame + 1];
			block[blkIdx++] = (float)reconsFrame[rowInFrame + 2];
			block[blkIdx++] = (float)reconsFrame[rowInFrame + 3];
			rowInFrame += codingOption.FrameWidth;
		}

		SMF blkSMF = smf[i];
		int smfIdx = 0;
		float sum = 0.0;
		for(int m=blkSMF.searchTop; m<=blkSMF.searchBottom; m++){
			for(int n=blkSMF.searchLeft; n<=blkSMF.searchRight; n++){
				float SSE = 0.0;
			    blkIdx = 0;
				rowInFrame = m * codingOption.FrameWidth + n;
				for(int ii=0; ii<4; ii++){
					//int blkRow = ii << 2;
					//int rowInFrame = (m+ii) * codingOption.FrameWidth + n;
					for(int jj=0; jj<4; jj++){
						float diff = block[blkIdx++] - (float)sideInfoFrame[rowInFrame + jj];
						SSE += diff * diff;
					}
					rowInFrame += codingOption.FrameWidth;
				}
				
				//blkSMF.prob[smfIdx] = exp(-MU * SSE / 16.0);
				blkSMF.prob[smfIdx] = exp((float)-MU * SSE);
				sum += blkSMF.prob[smfIdx];
				
				smfIdx++;
			}
		}

		if(sum > 0.0){
			//normalize
			for(int m=0; m<smfIdx; m++)
				blkSMF.prob[m] = blkSMF.prob[m]/sum;
		}
		else
			blkSMF.prob[blkSMF.blkIdx] = 1.0;
		smf[i] = blkSMF;
	}
}

void update_si_ML(int *sideTransFrame, Trans *transFrame, int band, SMF *smf, SIR *sir){
	int searchRangeWidth = codingOption.FrameWidth - 4 + 1; //use for transFrame
	for(int i=0; i<sir->searchCandidate; i++){
		//for each block
		SearchInfo searchInfo = sir->searchInfo[i];
		int top = searchInfo.top;
		int left = searchInfo.left;
		
		float block[BLOCKSIZE44] = {0};
		SMF blkSMF = smf[i];
		int smfIdx = 0;
		int rowInFrame = blkSMF.searchTop * searchRangeWidth;
		for(int m=blkSMF.searchTop; m<=blkSMF.searchBottom; m++){
			//int rowInFrame = m * searchRangeWidth;
			for(int n=blkSMF.searchLeft; n<=blkSMF.searchRight; n++){
				int idxInFrame = rowInFrame + n;
				float prob = blkSMF.prob[smfIdx];
				for(int k=band; k<BLOCKSIZE44; k++){
					int index = zigzag[k];						
					block[index] += prob * (float)(transFrame[idxInFrame].block44[index]);
				}
				smfIdx++;
			}
			rowInFrame += searchRangeWidth;
		}
		
		for(int m=band; m<BLOCKSIZE44; m++){
			int index = zigzag[m];
			sideTransFrame[(top+(index>>2)) * codingOption.FrameWidth + left + (index&3)] = (int)block[index];
		}
	}
}

void updateSMF_OpenMP(SMF *smf, unsigned char *reconsFrame, unsigned char *sideInfoFrame, SIR *sir){

#pragma omp parallel for
	for(int i=0; i<sir->searchCandidate; i++){
		//for each smf
		SearchInfo searchInfo = sir->searchInfo[i];
		int top = searchInfo.top;
		int left = searchInfo.left;

		float  block[BLOCKSIZE44];
		int blkIdx = 0;
		int rowInFrame = top * codingOption.FrameWidth + left;
		for(int m=0; m<4; m++){
			//int blkRow = m << 2;
			//int rowInFrame = (top+m) * codingOption.FrameWidth + left;
			//for(int n=0; n<4; n++)
			block[blkIdx++] = (float)reconsFrame[rowInFrame];
			block[blkIdx++] = (float)reconsFrame[rowInFrame + 1];
			block[blkIdx++] = (float)reconsFrame[rowInFrame + 2];
			block[blkIdx++] = (float)reconsFrame[rowInFrame + 3];
			rowInFrame += codingOption.FrameWidth;
		}

		SMF blkSMF = smf[i];
		int smfIdx = 0;
		float sum = 0.0;
		for(int m=blkSMF.searchTop; m<=blkSMF.searchBottom; m++){
			for(int n=blkSMF.searchLeft; n<=blkSMF.searchRight; n++){
				float SSE = 0.0;
			    blkIdx = 0;
				rowInFrame = m * codingOption.FrameWidth + n;
				for(int ii=0; ii<4; ii++){
					//int blkRow = ii << 2;
					//int rowInFrame = (m+ii) * codingOption.FrameWidth + n;
					for(int jj=0; jj<4; jj++){
						float diff = block[blkIdx++] - (float)sideInfoFrame[rowInFrame + jj];
						SSE += diff * diff;
					}
					rowInFrame += codingOption.FrameWidth;
				}
				
				//blkSMF.prob[smfIdx] = exp(-MU * SSE / 16.0);
				blkSMF.prob[smfIdx] = exp((float)-MU * SSE);
				sum += blkSMF.prob[smfIdx];
				
				smfIdx++;
			}
		}

		if(sum > 0.0){
			//normalize
			for(int m=0; m<smfIdx; m++)
				blkSMF.prob[m] = blkSMF.prob[m]/sum;
		}
		else
			blkSMF.prob[blkSMF.blkIdx] = 1.0;
		smf[i] = blkSMF;
	}
}

void update_si_ML_OpenMP(int *sideTransFrame, Trans *transFrame, int band, SMF *smf, SIR *sir){
	int searchRangeWidth = codingOption.FrameWidth - 4 + 1; //use for transFrame
#pragma omp parallel for
	for(int i=0; i<sir->searchCandidate; i++){
		//for each block
		SearchInfo searchInfo = sir->searchInfo[i];
		int top = searchInfo.top;
		int left = searchInfo.left;
		
		float block[BLOCKSIZE44] = {0};
		SMF blkSMF = smf[i];
		int smfIdx = 0;
		int rowInFrame = blkSMF.searchTop * searchRangeWidth;
		for(int m=blkSMF.searchTop; m<=blkSMF.searchBottom; m++){
			//int rowInFrame = m * searchRangeWidth;
			for(int n=blkSMF.searchLeft; n<=blkSMF.searchRight; n++){
				int idxInFrame = rowInFrame + n;
				float prob = blkSMF.prob[smfIdx];
				for(int k=band; k<BLOCKSIZE44; k++){
					int index = zigzag[k];						
					block[index] += prob * (float)(transFrame[idxInFrame].block44[index]);
				}
				smfIdx++;
			}
			rowInFrame += searchRangeWidth;
		}
		
		for(int m=band; m<BLOCKSIZE44; m++){
			int index = zigzag[m];
			sideTransFrame[(top+(index>>2)) * codingOption.FrameWidth + left + (index&3)] = (int)block[index];
		}
	}
}

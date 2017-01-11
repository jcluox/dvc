#include <stdlib.h>
#include <math.h>
#include "transform.h"
#include "omp.h"

double postScale[BLOCKSIZE44];
double preScale[BLOCKSIZE44];
//for Intra block
int *qp_per_matrix;
int *qp_rem_matrix;

void initTransform(){
	double a = 0.5;
	double b = sqrt(0.4);
	double a2 = 0.25, b2 = 0.4, ab = a*b;

	postScale[0] = a2;
	postScale[1] = ab/2;
	postScale[2] = a2;
	postScale[3] = ab/2;
	postScale[4] = ab/2;
	postScale[5] = b2/4;
	postScale[6] = ab/2;
	postScale[7] = b2/4;
	postScale[8] = a2;
	postScale[9] = ab/2;
	postScale[10] = a2;
	postScale[11] = ab/2;
	postScale[12] = ab/2;
	postScale[13] = b2/4;
	postScale[14] = ab/2;
	postScale[15] = b2/4;

	preScale[0] = a2;
	preScale[1] = ab;
	preScale[2] = a2;
	preScale[3] = ab;
	preScale[4] = ab;
	preScale[5] = b2;
	preScale[6] = ab;
	preScale[7] = b2;
	preScale[8] = a2;
	preScale[9] = ab;
	preScale[10] = a2;
	preScale[11] = ab;
	preScale[12] = ab;
	preScale[13] = b2;
	preScale[14] = ab;
	preScale[15] = b2;

	for(int i=0; i<BLOCKSIZE44; i++)
		preScale[i] *= 64.0;

	//for Intra block
	qp_per_matrix = (int*)malloc(sizeof(int)*(MAX_QP+1));
    qp_rem_matrix = (int*)malloc(sizeof(int)*(MAX_QP+1));
    for(int i=0;i<(MAX_QP+1);i++){
        qp_rem_matrix[i] = i % 6;
        qp_per_matrix[i] = i / 6;
    }
}

void forward44(int *blk){
	int tmpblk[BLOCKSIZE44];
	//tmp = C*X
	tmpblk[0] = blk[0] + blk[4] + blk[8] + blk[12];
	tmpblk[4] = (blk[0]<<1) + blk[4] - blk[8] - (blk[12]<<1);
	tmpblk[8] = blk[0] - blk[4] - blk[8] + blk[12];
	tmpblk[12] = blk[0] - (blk[4]<<1) + (blk[8]<<1) - blk[12];
	tmpblk[1] = blk[1] + blk[5] + blk[9] + blk[13];
	tmpblk[5] = (blk[1]<<1) + blk[5] - blk[9] - (blk[13]<<1);
	tmpblk[9] = blk[1] - blk[5] - blk[9] + blk[13];
	tmpblk[13] = blk[1] - (blk[5]<<1) + (blk[9]<<1) - blk[13];
	tmpblk[2] = blk[2] + blk[6] + blk[10] + blk[14];
	tmpblk[6] = (blk[2]<<1) + blk[6] - blk[10] - (blk[14]<<1);
	tmpblk[10] = blk[2] - blk[6] - blk[10] + blk[14];
	tmpblk[14] = blk[2] - (blk[6]<<1) + (blk[10]<<1) - blk[14];
	tmpblk[3] = blk[3] + blk[7] + blk[11] + blk[15];
	tmpblk[7] = (blk[3]<<1) + blk[7] - blk[11] - (blk[15]<<1);
	tmpblk[11] = blk[3] - blk[7] - blk[11] + blk[15];
	tmpblk[15] = blk[3] - (blk[7]<<1) + (blk[11]<<1) - blk[15];

	//C*X*C' = tmp*C'
	blk[0] = tmpblk[0] + tmpblk[1] + tmpblk[2] + tmpblk[3];
	blk[1] = (tmpblk[0]<<1) + tmpblk[1] - tmpblk[2] - (tmpblk[3]<<1);
	blk[2] = tmpblk[0] - tmpblk[1] - tmpblk[2] + tmpblk[3];
	blk[3] = tmpblk[0] - (tmpblk[1]<<1) + (tmpblk[2]<<1) - tmpblk[3];
	blk[4] = tmpblk[4] + tmpblk[5] + tmpblk[6] + tmpblk[7];
	blk[5] = (tmpblk[4]<<1) + tmpblk[5] - tmpblk[6] - (tmpblk[7]<<1);
	blk[6] = tmpblk[4] - tmpblk[5] - tmpblk[6] + tmpblk[7];
	blk[7] = tmpblk[4] - (tmpblk[5]<<1) + (tmpblk[6]<<1) - tmpblk[7];
	blk[8] = tmpblk[8] + tmpblk[9] + tmpblk[10] + tmpblk[11];
	blk[9] = (tmpblk[8]<<1) + tmpblk[9] - tmpblk[10] - (tmpblk[11]<<1);
	blk[10] = tmpblk[8] - tmpblk[9] - tmpblk[10] + tmpblk[11];
	blk[11] = tmpblk[8] - (tmpblk[9]<<1) + (tmpblk[10]<<1) - tmpblk[11];
	blk[12] = tmpblk[12] + tmpblk[13] + tmpblk[14] + tmpblk[15];
	blk[13] = (tmpblk[12]<<1) + tmpblk[13] - tmpblk[14] - (tmpblk[15]<<1);
	blk[14] = tmpblk[12] - tmpblk[13] - tmpblk[14] + tmpblk[15];
	blk[15] = tmpblk[12] - (tmpblk[13]<<1) + (tmpblk[14]<<1) - tmpblk[15];

	//C*X*C'.*postScale
	for(int i=0; i<BLOCKSIZE44; i++)
		blk[i] = ROUND( ((double)blk[i])*postScale[i] );
}

void inverse44(int *blk){
	int tmpblk[BLOCKSIZE44];
	//preScale
	for(int i=0; i<BLOCKSIZE44; i++)
		blk[i] = ROUND( ((double)blk[i])*preScale[i] );

	//tmp = IC*X
	tmpblk[0] = blk[0] + blk[4] + blk[8] + (blk[12]>>1);
	tmpblk[4] = blk[0] + (blk[4]>>1) - blk[8] - blk[12];
	tmpblk[8] = blk[0] - (blk[4]>>1) - blk[8] + blk[12];
	tmpblk[12] = blk[0] - blk[4] + blk[8] - (blk[12]>>1);
	tmpblk[1] = blk[1] + blk[5] + blk[9] + (blk[13]>>1);
	tmpblk[5] = blk[1] + (blk[5]>>1) - blk[9] - blk[13];
	tmpblk[9] = blk[1] - (blk[5]>>1) - blk[9] + blk[13];
	tmpblk[13] = blk[1] - blk[5] + blk[9] - (blk[13]>>1);
	tmpblk[2] = blk[2] + blk[6] + blk[10] + (blk[14]>>1);
	tmpblk[6] = blk[2] + (blk[6]>>1) - blk[10] - blk[14];
	tmpblk[10] = blk[2] - (blk[6]>>1) - blk[10] + blk[14];
	tmpblk[14] = blk[2] - blk[6] + blk[10] - (blk[14]>>1);
	tmpblk[3] = blk[3] + blk[7] + blk[11] + (blk[15]>>1);
	tmpblk[7] = blk[3] + (blk[7]>>1) - blk[11] - blk[15];
	tmpblk[11] = blk[3] - (blk[7]>>1) - blk[11] + blk[15];
	tmpblk[15] = blk[3] - blk[7] + blk[11] - (blk[15]>>1);

	//IC*X*IC' = tmp*IC'
	blk[0] = tmpblk[0] + tmpblk[1] + tmpblk[2] + (tmpblk[3]>>1);
	blk[1] = tmpblk[0] + (tmpblk[1]>>1) - tmpblk[2] - tmpblk[3];
	blk[2] = tmpblk[0] - (tmpblk[1]>>1) - tmpblk[2] + tmpblk[3];
	blk[3] = tmpblk[0] - tmpblk[1] + tmpblk[2] - (tmpblk[3]>>1);
	blk[4] = tmpblk[4] + tmpblk[5] + tmpblk[6] + (tmpblk[7]>>1);
	blk[5] = tmpblk[4] + (tmpblk[5]>>1) - tmpblk[6] - tmpblk[7];
	blk[6] = tmpblk[4] - (tmpblk[5]>>1) - tmpblk[6] + tmpblk[7];
	blk[7] = tmpblk[4] - tmpblk[5] + tmpblk[6] - (tmpblk[7]>>1);
	blk[8] = tmpblk[8] + tmpblk[9] + tmpblk[10] + (tmpblk[11]>>1);
	blk[9] = tmpblk[8] + (tmpblk[9]>>1) - tmpblk[10] - tmpblk[11];
	blk[10] = tmpblk[8] - (tmpblk[9]>>1) - tmpblk[10] + tmpblk[11];
	blk[11] = tmpblk[8] - tmpblk[9] + tmpblk[10] - (tmpblk[11]>>1);
	blk[12] = tmpblk[12] + tmpblk[13] + tmpblk[14] + (tmpblk[15]>>1);
	blk[13] = tmpblk[12] + (tmpblk[13]>>1) - tmpblk[14] - tmpblk[15];
	blk[14] = tmpblk[12] - (tmpblk[13]>>1) - tmpblk[14] + tmpblk[15];
	blk[15] = tmpblk[12] - tmpblk[13] + tmpblk[14] - (tmpblk[15]>>1);

	//divide by 64
	for(int i=0; i<BLOCKSIZE44; i++)
		blk[i] >>= 6;
}

void forwardTransform(int *oriFrame, int *transFrame){
	for(int i=0; i<Block44Height; i++){
		int top = i << 2;
		for(int j=0; j<Block44Width; j++){
			//for each block
			int left = j << 2;

			//forward transform
			int initRow = top * codingOption.FrameWidth + left;
			int block[BLOCKSIZE44];
			int blkIdx = 0;
			int rowInFrame = initRow;
			for(int m=0; m<4; m++){
				//for(int n=0; n<4; n++)
				block[blkIdx++] = oriFrame[rowInFrame];
				block[blkIdx++] = oriFrame[rowInFrame + 1];
				block[blkIdx++] = oriFrame[rowInFrame + 2];
				block[blkIdx++] = oriFrame[rowInFrame + 3];
				rowInFrame += codingOption.FrameWidth;
			}			
				
			//transform
			forward44(block);

			rowInFrame = initRow;
			blkIdx = 0;
			for(int m=0; m<4; m++){
				//for(int n=0; n<4; n++)
				transFrame[rowInFrame] = block[blkIdx++];
				transFrame[rowInFrame + 1] = block[blkIdx++];
				transFrame[rowInFrame + 2] = block[blkIdx++];
				transFrame[rowInFrame + 3] = block[blkIdx++];
				rowInFrame += codingOption.FrameWidth;
			}		

		}
	}
}

void inverseTransform_OpenMP(int *transFrame, unsigned char *reconsFrame, SIR *sir){
	#pragma omp parallel for
	for(int i=0; i<sir->searchCandidate; i++){
		//for each WZ block
		int top = sir->searchInfo[i].top;
		int left = sir->searchInfo[i].left;

		//inverse transform
		int initRow = top * codingOption.FrameWidth + left;
		int block[BLOCKSIZE44];
		int blkIdx = 0;
		int rowInFrame = initRow;
		for(int m=0; m<4; m++){
			//for(int n=0; n<4; n++)
			block[blkIdx++] = transFrame[rowInFrame];
			block[blkIdx++] = transFrame[rowInFrame + 1];
			block[blkIdx++] = transFrame[rowInFrame + 2];
			block[blkIdx++] = transFrame[rowInFrame + 3];
			rowInFrame += codingOption.FrameWidth;
		}

		//transform
		inverse44(block);
		
		blkIdx = 0;
		rowInFrame = initRow;
		for(int m=0; m<4; m++){
			for(int n=0; n<4; n++){
				int coeff = block[blkIdx++];
				if(coeff < 0)
					reconsFrame[rowInFrame + n] = (unsigned char)0;
				else if(coeff > 255)
					reconsFrame[rowInFrame + n] = (unsigned char)255;
				else
					reconsFrame[rowInFrame + n] = (unsigned char)coeff;
			}		
			rowInFrame += codingOption.FrameWidth;
		}

	}	
}

void inverseTransform(int *transFrame, unsigned char *reconsFrame, SIR *sir){
	for(int i=0; i<sir->searchCandidate; i++){
		//for each WZ block
		int top = sir->searchInfo[i].top;
		int left = sir->searchInfo[i].left;

		//inverse transform
		int initRow = top * codingOption.FrameWidth + left;
		int block[BLOCKSIZE44];
		int blkIdx = 0;
		int rowInFrame = initRow;
		for(int m=0; m<4; m++){
			//for(int n=0; n<4; n++)
			block[blkIdx++] = transFrame[rowInFrame];
			block[blkIdx++] = transFrame[rowInFrame + 1];
			block[blkIdx++] = transFrame[rowInFrame + 2];
			block[blkIdx++] = transFrame[rowInFrame + 3];
			rowInFrame += codingOption.FrameWidth;
		}

		//transform
		inverse44(block);
		
		blkIdx = 0;
		rowInFrame = initRow;
		for(int m=0; m<4; m++){
			for(int n=0; n<4; n++){
				int coeff = block[blkIdx++];
				if(coeff < 0)
					reconsFrame[rowInFrame + n] = (unsigned char)0;
				else if(coeff > 255)
					reconsFrame[rowInFrame + n] = (unsigned char)255;
				else
					reconsFrame[rowInFrame + n] = (unsigned char)coeff;
			}		
			rowInFrame += codingOption.FrameWidth;
		}

	}	
}

void overCompleteTransform(unsigned char *pixelFrame, Trans *transFrame, int searchRangeBottom, int searchRangeRight){
	int transIdx = 0;
	for(int top=0; top<=searchRangeBottom; top++){
		for(int left=0; left<=searchRangeRight; left++){
			int blkIdx = 0;
			int frameRow = top * codingOption.FrameWidth + left;
			for(int m=0; m<4; m++){
				//int frameRow = (top + m) * codingOption.FrameWidth + left;
				//for(int n=0; n<4; n++){
				transFrame[transIdx].block44[blkIdx++] = pixelFrame[frameRow];
				transFrame[transIdx].block44[blkIdx++] = pixelFrame[frameRow + 1];
				transFrame[transIdx].block44[blkIdx++] = pixelFrame[frameRow + 2];
				transFrame[transIdx].block44[blkIdx++] = pixelFrame[frameRow + 3];
				//}
				frameRow += codingOption.FrameWidth;
			}

			//transform
			forward44(transFrame[transIdx].block44);

			transIdx++;
		}
	}
}

void overCompleteTransform_OpenMP(unsigned char *pixelFrame, Trans *transFrame, int searchRangeBottom, int searchRangeRight){
	int searchRangeWidth = codingOption.FrameWidth - 4 + 1; //used for overcomplete_trans
#pragma omp parallel for
	for(int top=0; top<=searchRangeBottom; top++){
		int transRow = top * searchRangeWidth;
		for(int left=0; left<=searchRangeRight; left++){
			int transIdx = transRow + left;
			int blkIdx = 0;
			int frameRow = top * codingOption.FrameWidth + left;
			for(int m=0; m<4; m++){
				//int frameRow = (top + m) * codingOption.FrameWidth + left;
				//for(int n=0; n<4; n++){
				transFrame[transIdx].block44[blkIdx++] = pixelFrame[frameRow];
				transFrame[transIdx].block44[blkIdx++] = pixelFrame[frameRow + 1];
				transFrame[transIdx].block44[blkIdx++] = pixelFrame[frameRow + 2];
				transFrame[transIdx].block44[blkIdx++] = pixelFrame[frameRow + 3];
				//}
				frameRow += codingOption.FrameWidth;
			}

			//transform
			forward44(transFrame[transIdx].block44);
		}
	}
}
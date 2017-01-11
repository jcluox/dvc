#include <math.h>
#include "global.h"
#include "quantization.h"

//use for quantization
int *Qtable;
int *QtableBits;
int QbitsNumPerBlk = 0;	//number of bits per block
int ACLevelShift[BLOCKSIZE44] = {0};	//AC level shift (let all coefficients >= 0)
int stepSize[BLOCKSIZE44];
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

//perform quantization to transFrame
void quantize(int *transFrame, EncodedFrame *encodedFrame){
	//calculate quantization step size
	//DC range: [0, 1024)
	stepSize[0] = 1024 / Qtable[0];
	//AC dynamic range
	for(int i=1; i<BLOCKSIZE44; i++){
		if(Qtable[i] > 0)
			stepSize[i] = (int)ceil((double)( (encodedFrame->ACRange[i]+1)<<1 ) / (double)Qtable[i]); // ceil( 2*|Valmax| / level )
		else
			stepSize[i] = 0;
	}

	int blkIdx = 0;
	for(int i=0; i<Block88Height; i++){
		int top = i << 3;
		for(int j=0; j<Block88Width; j++){
			//for each block
			if(encodedFrame->skipBlockFlag[blkIdx++] == 0){
				int left = j << 3;

				int block[BLOCKSIZE44];
				for(int ii=0; ii<8; ii+=4){
					for(int jj=0; jj<8; jj+=4){
						
						for(int m=0; m<4; m++){
							int idx1 = m << 2;
							int idx2 = (top + ii + m) * (codingOption.FrameWidth) + left + jj;
							for(int n=0; n<4; n++)
								block[idx1 + n] = transFrame[idx2 + n];			
						}				
						
						//quantize
						quantize44(block, stepSize);

						for(int m=0; m<4; m++){
							int idx1 = m << 2;
							int idx2 = (top + ii + m) * (codingOption.FrameWidth) + left + jj;
							for(int n=0; n<4; n++)
								transFrame[idx2 + n] = block[idx1 + n];
						}

					}
				}
			}
		}
	}
}
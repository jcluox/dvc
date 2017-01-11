#include "global.h"
#include "reconstruction.h"

void putBitplaneToCoeff(unsigned char *decodedBitplane, int *reconsTransFrame, int indexInBlk, unsigned char *BlockModeFlag){
	
	int indexInBlkRow = indexInBlk >> 2; //indexInblk / 4
	int indexInBlkCol = indexInBlk & 3;	//indexInblk % 4

	int blkIdx = 0;	//used for decodedBitplane
	for(int i=0; i<Block44Height; i++){
		int rowInFrame = ((i<<2) + indexInBlkRow) * codingOption.FrameWidth;
		int blk88row = (i>>1) * Block88Width;
		for(int j=0; j<Block44Width; j++){
			//for each block
			if(BlockModeFlag[blk88row + (j>>1)] == WZ_B){
				int indexInFrame = rowInFrame + (j<<2) + indexInBlkCol;	//used for reconsTransFrame, sideTransFrame
				reconsTransFrame[indexInFrame] = (reconsTransFrame[indexInFrame] << 1) + (int)decodedBitplane[blkIdx];
			}
			blkIdx++;
		}
	}
}

void copyNonSendCeff(int *reconsTransFrame, int *sideTransFrame, unsigned char *BlockModeFlag){

	for(int i=0; i<Block44Height; i++){
		int top = i << 2;
		int blk88row = (i>>1) * Block88Width;
		for(int j=0; j<Block44Width; j++){
			//for each block
			int left = j << 2;

			if(BlockModeFlag[blk88row + (j>>1)] == WZ_B){
				for(int m=0; m<4; m++){
					int rowInBlk = m << 2;
					int rowInFrame = (top+m) * codingOption.FrameWidth + left;
					for(int n=0; n<4; n++){
						if(Qtable[rowInBlk + n] <= 0)							
							reconsTransFrame[rowInFrame + n] = sideTransFrame[rowInFrame + n];	//non-send band
					}
				}
			}
		}
	}
}

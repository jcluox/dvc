#include "global.h"

void copySkipBlkToRecons(unsigned char *reconsFrame, unsigned char *refFrame, unsigned char *BlockModeFlag){
	int blk88Idx = 0;
	for(int i=0; i<Block88Height; i++){
		int top = i << 3;
		for(int j=0; j<Block88Width; j++){
			//for each block
			int left = j << 3;

			if(BlockModeFlag[blk88Idx++] == SKIP_B){
				#pragma unroll 8
				for(int m=0; m<8; m++){
					int rowInFrame = (top+m) * codingOption.FrameWidth + left;
					#pragma unroll 8
					for(int n=0; n<8; n++){
						reconsFrame[rowInFrame + n] = refFrame[rowInFrame + n];
					}
				}
			}
		}
	}
}
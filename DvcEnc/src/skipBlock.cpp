#include <string.h>
#include <stdlib.h>
#include "skipBlock.h"
#include "quantization.h"
#include "sJBIG.h"

void findSkipBlock(unsigned char *oriFrame, unsigned char *refFrame, EncodedFrame *encodedFrame){

	/*int minStep = stepSize[0];
	for(int i=1;i<BLOCKSIZE44; i++){
		if(stepSize[i] < minStep)
			minStep = stepSize[0];
	}
	double skipThreshold = (double)(minStep * minStep) / (double)BLOCKSIZE88;
*/
	int blkIdx = 0;
	int skipBlock = 0;
	for(int i=0; i<Block88Height; i++){
		int top = i << 3;
		for(int j=0; j<Block88Width; j++){
			//for each block
			int left = j << 3;

			//find skip block
			double mse = 0.0;
			for(int m=0; m<8; m++){
				int idx = (top+m) * (codingOption.FrameWidth) + left;
				for(int n=0; n<8; n++){
					double dst = (double)oriFrame[idx + n] - (double)refFrame[idx + n];				
					mse += dst * dst;
				}		
			}
			mse = mse / BLOCKSIZE88;

			//if(mse < skipThreshold){
			if(mse < SKIPTHRESHOLD){
				//skip block			
				encodedFrame->skipBlockFlag[blkIdx] = 1;
				skipBlock++;
			}
			blkIdx++;
		}
	}
	//printf("skipBlock:%d\n",skipBlock);
	//system("pause");

	//cancel skip mode of this frame if number of skip blocks is too small
	unsigned char *bitmap_skip_temp = encodedFrame->skipBlockFlag;
    int new_width = Block88Width;
    int new_height = Block88Height;
    int bitmap_byte = 0;
    unsigned char *jbigdata = BitToBitmap(bitmap_skip_temp, new_width, new_height, &bitmap_byte);
    
    unsigned char jbgh[20];
    int enc_one_bytes = simple_JBIG_enc(NULL,jbigdata,new_width,new_height,jbgh,new_height,0);
    //printf("enc_one_byte = %d\n",enc_one_bytes);    
    int skipMin = (int)((double)(enc_one_bytes<<3) * U / (double)QbitsNumPerBlk);
    //printf("SkipMin = %d\n",SkipMin);
    free(jbigdata);

    //cancel skip mode of this frame if number of skip blocks is too small
    if(skipBlock < skipMin){
        memset(encodedFrame->skipBlockFlag, 0, sizeof(unsigned char) * Block88Num);
    }else{
        encodedFrame->skipMode = 1;
    }
}

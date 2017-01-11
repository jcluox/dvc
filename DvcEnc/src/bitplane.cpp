#include "bitplane.h"
#include "quantization.h"

void extractBitplane(int *transFrame, unsigned char **bitplanes, EncodedFrame *encodedFrame){
	int bit = 0;	//from bitplane MSB to LSB ( length: block number )
	for(int i=0; i<Block44Height; i++){
		int top = i << 2;
		int blk88row = (i>>1) * Block88Width;
		for(int j=0; j<Block44Width; j++){
			//for each 4x4 block
			int left = j << 2;
			int blk88Idx = blk88row + (j>>1);
			if(encodedFrame->skipBlockFlag[blk88Idx] == 0 && encodedFrame->intraBlockFlag[blk88Idx] == 0){
				//zig-zag order
				int band = 0;	//from DC band
				int index = zigzag[band];	//index in block (zig-zag order)
				int Qbits = QtableBits[index];	//how many bits of this band
				int k = 0;	//index of bitplane (length: QbitsNumPerBlk)
				while(Qbits > 0){
					int coefficient = transFrame[(top + (index>>2)) * (codingOption.FrameWidth) + left + (index&3)];	//index>>2 -> index/4, index&3 -> index%4
					//from coefficient MSB to LSB
					for(int n=Qbits-1; n>=0; n--){
						bitplanes[k++][bit] = (coefficient>>n) & 1;
					}

					band++;
					index = zigzag[band];			
					Qbits = QtableBits[index];			
				}					
			}
			else{
				for(int k=0; k<QbitsNumPerBlk; k++)
					bitplanes[k][bit] = 0;
			}
			bit++;
		}
	}

}
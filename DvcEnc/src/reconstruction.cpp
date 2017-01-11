#include <math.h>
#include "reconstruction.h"
#include "quantization.h"

void reconstruction(int *reconsTransFrame, int *quantizedFrame, int *sideTransFrame, double *alpha, EncodedFrame *encodedFrame){
    
    for(int i=0; i<Block44Height; i++){
        int top = i << 2;
        int blk88row = (i>>1) * Block88Width;
        for(int j=0; j<Block44Width; j++){
            //for each 4x4 block
			int blk88Idx = blk88row + (j>>1);
            if(encodedFrame->skipBlockFlag[blk88Idx] == 0 && encodedFrame->intraBlockFlag[blk88Idx] == 0){
				//WZ block
                int left = j << 2;

                //zig-zag order
                //DC band reconstruction
                int coeffIdx = top * codingOption.FrameWidth + left;
                int stepsize = stepSize[0];
                int qIdx = quantizedFrame[coeffIdx];
                int val_lower = qIdx * stepsize;
                int val_upper = (qIdx + 1) * stepsize;
                int sideCeff = sideTransFrame[coeffIdx];
                double alphaCeff = alpha[coeffIdx];
                if(alphaCeff <= 0.0001){
                    reconsTransFrame[coeffIdx] = (val_lower + val_upper) / 2;
                }
                else{
                    if(sideCeff < val_lower){
                        reconsTransFrame[coeffIdx] = val_lower + ROUND( 1.0 / alphaCeff + (double)stepsize / (1.0 - exp(alphaCeff * (double)stepsize)) );
                    }
                    else if(sideCeff >= val_upper){
                        reconsTransFrame[coeffIdx] = val_upper - ROUND( 1.0 / alphaCeff + (double)stepsize / (1.0 - exp(alphaCeff * (double)stepsize)) );
                    }
                    else{
                        double r = sideCeff - val_lower;
                        double delta = val_upper - sideCeff;
                        double exp_alpha_r = exp(-alphaCeff * r);
                        double exp_alpha_delta = exp(-alphaCeff * delta);
                        reconsTransFrame[coeffIdx] = sideCeff + ROUND(
                                ( (r + 1.0 / alphaCeff) * exp_alpha_r - (delta + 1.0 / alphaCeff) * exp_alpha_delta ) / ( 2.0 - (exp_alpha_r + exp_alpha_delta) )
                                );
                    }
                }
                //AC band reconstruction
				for(int index=1; index<BLOCKSIZE44; index++){
                    //for each band
					coeffIdx = (top+(index>>2)) * (codingOption.FrameWidth) + left+ (index&3);  //index>>2 -> index/4, index&3 -> index%4
					sideCeff = sideTransFrame[coeffIdx];
					if(QtableBits[index] > 0){                    
						stepsize = stepSize[index];
						qIdx = quantizedFrame[coeffIdx] - ACLevelShift[index];
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
	       
						alphaCeff = alpha[coeffIdx];
						if(alphaCeff <= 0.0001){
							reconsTransFrame[coeffIdx] = (val_lower + val_upper) / 2;
						}
						else{
							if(sideCeff < val_lower){
								reconsTransFrame[coeffIdx] = val_lower + ROUND( 1.0 / alphaCeff + (double)stepsize / (1.0 - exp(alphaCeff * (double)stepsize)) );
							}
							else if(sideCeff >= val_upper){
								reconsTransFrame[coeffIdx] = val_upper - ROUND( 1.0 / alphaCeff + (double)stepsize / (1.0 - exp(alphaCeff * (double)stepsize)) );
							}
							else{
								double r = sideCeff - val_lower;
								double delta = val_upper - sideCeff;
								double exp_alpha_r = exp(-alphaCeff * r);
								double exp_alpha_delta = exp(-alphaCeff * delta);
								reconsTransFrame[coeffIdx] = sideCeff + ROUND(
										( (r + 1.0 / alphaCeff) * exp_alpha_r - (delta + 1.0 / alphaCeff) * exp_alpha_delta ) / ( 2.0 - (exp_alpha_r + exp_alpha_delta) )
										);
							}
						}

					}
					else{
						//copy from side info
						reconsTransFrame[coeffIdx] = sideCeff;
					}
                }
            }

        }
    }
}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "modeSelection.h"
#include "intra.h"
#include "transform.h"
#include "quantization.h"
#include "sJBIG.h"
#include "intraBitstream.h"

Intra_Param intrap;

void intra44selection(int block44pos, EncodedFrame *encodedFrame, unsigned char *sideInfoFrame, int frameIdx, double *total_RIW,double *total_RWZ){
    double RIW, RWZ[16];
 
    /* intra rate estimation */                
    if(Recal_Intra){
        RIW = encodedFrame->Rspent[block44pos];
    }else{
        RIW = (double)((int)((intrap.thita)*(encodedFrame->Z[block44pos]) + (intrap.kappa)*(encodedFrame->E[block44pos]) + 0.5)); 
    }
    (*total_RIW) += RIW;

    /* WZ rate estimation */
    // find MAD
	int b_x = block44pos % Block44Width;
    int b_y = block44pos / Block44Width;
    int MAD = 0;
    for(int i=0;i<4;i++){
        int pos_y = ((b_y << 2) + i) * codingOption.FrameWidth + (b_x << 2);
        for(int j=0;j<4;j++){
            MAD += abs(oriFrames[frameIdx][pos_y+j] - sideInfoFrame[pos_y+j]);
        }
    }
    MAD = (MAD+8) >> 4;

    // find alpha
    double alpha[BLOCKSIZE44];
    for(int i=0;i<4;i++){
        int pos_y = ((b_y << 2) + i) * codingOption.FrameWidth + (b_x << 2);
        for(int j=0;j<4;j++){
            alpha[(i<<2)+j] = 1.0/ ((sqrt(Kmatrix[(i<<2)+j]))*(double)MAD);
            alpha_perframe[pos_y+j] = alpha[(i<<2)+j];
        }
    }

    // calculate WZ rate estimation
    double RWZ_sum = 0.0;
    double gama = GAMA;
    
    for(int i=0;i<4;i++){
        for(int j=0;j<4;j++){
            int delta = stepSize[(i<<2)+j];
            int idx = (i << 2) + j;
            double tempalpha = alpha[(i<<2)+j];

            if(delta > 0 && delta < (4.0/tempalpha)){                
                double temp = 4.0 * gama / (tempalpha * (double)delta);
                if(temp < 1.0)
                    RWZ[idx] = 0.0;
                else
                    RWZ[idx] = log(temp);
            }else{
                RWZ[idx] = 0.0;
            }
            RWZ_sum += (double)((int)(RWZ[idx]+0.5));
        }
    }

    (*total_RWZ) += RWZ_sum;
}

void recal_para(int blocknum, EncodedFrame *encodedFrame){
    double *E = encodedFrame->E;
    double *Z = encodedFrame->Z;
    int *Rspent = encodedFrame->Rspent;

    double Ei2=0.0,Zi2=0.0,ZiEi=0.0;
    double ZiR=0.0,EiR=0.0;

    for(int i=0;i<blocknum;i++){
        Ei2 += E[i]*E[i];
        Zi2 += Z[i]*Z[i];
        ZiEi += Z[i]*E[i];
        ZiR += Z[i]*(double)Rspent[i];
        EiR += E[i]*(double)Rspent[i];
    }

    double denom =0.0;

    denom = (Ei2 * Zi2) - (ZiEi * ZiEi);
    intrap.thita = (Ei2 * ZiR - ZiEi * EiR) / denom;
    intrap.kappa = (Zi2 * EiR - ZiEi * ZiR) / denom;

}

void intra_selection(EncodedFrame *encodedFrame, unsigned char *sideInfoFrame, int frameIdx){
    //double RIW = 0.0;
    //double RWZ = 0.0;
    unsigned char *intraBlockFlag = encodedFrames[frameIdx].intraBlockFlag;
    unsigned char *skipBlockFlag  = encodedFrames[frameIdx].skipBlockFlag;
    int qp_per = qp_per_matrix[codingOption.IntraQP];
    int qbits = Q_BITS + qp_per;
    //int qbits = (int)(15.0 + floor((double)codingOption.IntraQP / 6.0));
    int offset = (int)( ((double)(1<<qbits) / 3.0) + 0.5);    
    double Qstep = pow( 2.0, (((double)codingOption.IntraQP - 4) / 6.0));
    //printf("qbits = %d, offset = %d, Qstep = %lf\n",qbits,offset,Qstep);

    int rate = 0;
	
    for(int i=0; i<Block88Height; i++){
        for(int j=0; j<Block88Width; j++){
			//for each 8x8 block
			int block88pos = i * Block88Width + j;
            if(skipBlockFlag[block88pos] == 0){
                double RWZ_88 = 0.0;
                double RIW_88 = 0.0;

				// set binary decision map to 1 for intra calculation
				intraBlockFlag[block88pos] = 1;
                for(int m=0; m<2; m++){
                    int block44_idx_y = (i << 1) + m;
                    int block44_idx_x = (j << 1);
                    int idx = block44_idx_y * Block44Width + block44_idx_x;
                    for(int n=0; n<2; n++){
                        block44_idx_x += n;
                        int block44pos = idx + n;
                        //printf("block[%2d,%2d] (%d)\n",block44_idx_y,block44_idx_x,block44pos);
						
                        intra_prediction(frameIdx,block44_idx_x,block44_idx_y,encodedFrame,block44pos,qbits,offset,Qstep,0,Recal_Intra);

                        intra44selection(block44pos,encodedFrame,sideInfoFrame,frameIdx,&RIW_88,&RWZ_88);
                    }

                }

				//mode decision				
				if(RWZ_88 <= RIW_88)
					intraBlockFlag[block88pos] = 0;	// use WZ
                //RIW += RIW_88;
                //RWZ += RWZ_88;
            }
        }
	}

    if(Recal_Intra){
        recal_para(Block44Num, encodedFrame);
    }

    //printf("RWZ : %5.0lf\n",RWZ);
    /*if(Recal_Intra){
        printf("RIW(%5d)*: %5.0lf, RWZ : %5.0lf =>  ",rate,RIW,RWZ);
    }else{ 
        printf("RIW(%5d) : %5.0lf, RWZ : %5.0lf =>  ",rate,RIW,RWZ);
    }*/
}

void recal_decision(EncodedFrame *encodedFrame){
    // JBIG compression per frame
    int new_width;
    int new_height;
    int bitmap_byte = 0;
    unsigned char *jbigdata;
    
	int count_intra = 0;
	for(int i=0; i<Block88Num; i++)
		count_intra += encodedFrame->intraBlockFlag[i];

    new_width = Block88Width;
    new_height = Block88Height;
    jbigdata = BitToBitmap(encodedFrame->intraBlockFlag, new_width, new_height, &bitmap_byte);
    int enc_one_bytes = 0;
    unsigned char jbgh[20];
    enc_one_bytes = simple_JBIG_enc(NULL,jbigdata,new_width,new_height,jbgh,new_height,0);
    
	int intraMin = ((int)((double)(enc_one_bytes<<3) * U / (double)QbitsNumPerBlk));
	if(count_intra < intraMin){
        memset(encodedFrame->intraBlockFlag, 0, sizeof(unsigned char) * Block88Num);
        encodedFrame->intraCount = 0;
    }else{
        encodedFrame->intraMode = 1;
        encodedFrame->intraCount = count_intra;
    }


    if(count_intra >= ((Block88Num+1)>>1)){
        Recal_Intra = 1;
    }else{
        Recal_Intra = 0;
    }
    free(jbigdata);
}

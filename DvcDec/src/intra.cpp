#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

#include "intra.h"
#include "vlc.h"
#include "inlineFunc.h"
#include "transform.h"


const int dequant_coef[6][BLOCKSIZE44] = {
    {10, 13, 10, 13, 13, 16, 13, 16, 10, 13, 10, 13, 13, 16, 13, 16},
    {11, 14, 11, 14, 14, 18, 14, 18, 11, 14, 11, 14, 14, 18, 14, 18},
    {13, 16, 13, 16, 16, 20, 16, 20, 13, 16, 13, 16, 16, 20, 16, 20},
    {14, 18, 14, 18, 18, 23, 18, 23, 14, 18, 14, 18, 18, 23, 18, 23},
    {16, 20, 16, 20, 20, 25, 20, 25, 16, 20, 16, 20, 20, 25, 20, 25},
    {18, 23, 18, 23, 23, 29, 23, 29, 18, 23, 18, 23, 23, 29, 23, 29}
};

int bitdepth_luma = 8;


void printblock_i(int *block){

    for(int i=0;i<4;i++){
        for(int j=0;j<4;j++){
            printf("%2d ",block[(i<<2)+j]);
        }
        printf("\n");
    }
}


void inverse44_intra(int *blk){
    int tmpblk[BLOCKSIZE44];
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

void getNeighbour(int X,int Y,int refB,int xN, int yN, PixelPos *pix, unsigned char *BlockModeFlag){

    switch(refB){
        case 0:
            // A
            if(xN < 0){
                pix->available = 0;
            }else{
                if(BlockModeFlag[(Y>>1)*Block88Width + ((X-1)>>1)] == INTRA_B){
                    pix->available = 1;
                }else{
                    pix->available = 0;
                }
            }
            pix->pos_x = xN;
            pix->pos_y = yN;
            break;
        case 1:
            // B
            if(yN < 0){
                pix->available = 0;
            }else{
                if(BlockModeFlag[((Y-1)>>1)*Block88Width + (X>>1)] == INTRA_B){
                    pix->available = 1;
                }else{
                    pix->available = 0;
                }
            }
            pix->pos_x = xN;
            pix->pos_y = yN;
            break;
        default:
            pix->available = 0;
            pix->pos_x = 0;
            pix->pos_y = 0;
            break;
    }


}

void intra4x4_dc_pred(unsigned char *imgY, int X, int Y, unsigned char *cur_pred, unsigned char *BlockModeFlag){
    int s0 = 0;  
    unsigned int dc_pred_value = 1<<(bitdepth_luma - 1);
    PixelPos pix_a[4], pix_b;

    int block_available_up;
    int block_available_left;
    //unsigned char *imgY = Intra_reconFrames[frameIdx];
    unsigned char *img_temp; 
    //unsigned char *bitmapNow = bitmap[frameIdx];

    int ioff = (X<<2);
    int joff = (Y<<2);

    for (int i=0;i<4;i++){
        getNeighbour(X,Y,0,ioff - 1,joff + i,&pix_a[i],BlockModeFlag);
    }
    getNeighbour(X,Y,1,ioff,joff - 1,&pix_b,BlockModeFlag);

    block_available_left     = pix_a[0].available;
    block_available_up       = pix_b.available;

    // form predictor pels
    if (block_available_up){
        img_temp = &imgY[(pix_b.pos_y)*codingOption.FrameWidth + pix_b.pos_x];
        s0 += *(img_temp++);
        s0 += *(img_temp++);
        s0 += *(img_temp++);
        s0 += *(img_temp);
    }

    if (block_available_left){
        s0 += imgY[pix_a[0].pos_y*codingOption.FrameWidth + pix_a[0].pos_x];
        s0 += imgY[pix_a[1].pos_y*codingOption.FrameWidth + pix_a[1].pos_x];
        s0 += imgY[pix_a[2].pos_y*codingOption.FrameWidth + pix_a[2].pos_x];
        s0 += imgY[pix_a[3].pos_y*codingOption.FrameWidth + pix_a[3].pos_x];
    }

    if (block_available_up && block_available_left)
    {
        // no edge
        s0 = (s0 + 4)>>3;
    }
    else if (!block_available_up && block_available_left)
    {
        // upper edge
        s0 = (s0 + 2)>>2;
    }
    else if (block_available_up && !block_available_left)
    {
        // left edge
        s0 = (s0 + 2)>>2;
    }
    else //if (!block_available_up && !block_available_left)
    {
        // top left corner, nothing to predict from
        s0 = dc_pred_value;
    }

    for (int i=0;i<BLOCKSIZE44;i++){
        // store DC prediction
        *cur_pred++ = (unsigned char) s0;
    }
}



void read_one_block(int block[BLOCKSIZE44], EncodedFrame *encodedFrame, int X, int Y, double Qstep){
    int numcoeff = 0;
    int levarr[BLOCKSIZE44],runarr[BLOCKSIZE44];
    int coeff[BLOCKSIZE44];
    int qp_rem = qp_rem_matrix[codingOption.IntraQP];
    const int *invlevelscale = dequant_coef[qp_rem];
    int qp_per = qp_per_matrix[codingOption.IntraQP];

    memset(levarr,0,sizeof(int)*BLOCKSIZE44);
    memset(runarr,0,sizeof(int)*BLOCKSIZE44);
    memset(coeff,0,sizeof(int)*BLOCKSIZE44);

    // read CAVLC coefficient
    readCoeff4x4_CAVLC(X, Y, levarr, runarr, &numcoeff, encodedFrame);

    // replace the position
    int pos=0;
    for (int k = 0; k < numcoeff; k++){
        if(levarr[k]!=0){
            pos += runarr[k];
            coeff[zigzag[pos]] = levarr[k];
            pos++;
        } 
    }

    // dequantization
    for(int i=0;i<BLOCKSIZE44;i++){
        block[i] = ROUND((double)coeff[i] * Qstep * preScale[i]);
        //block[i] = rshift_rnd_sf(((coeff[i] * (invlevelscale[i] << 4)) << qp_per), 4);
    }

}

void SampleRecon(unsigned char *reconsFrame, int *block, unsigned char *cur_pred, int X, int Y){
    //unsigned char *reconsFrame = (unsigned char*)Intra_reconFrames[frameIdx];
    static unsigned char *imgOrg, *imgPred;
    int *imgDiff;
    int top = Y << 2;
    int left = X << 2;

    imgPred = cur_pred;
    imgDiff = block;


    for(int j=0;j<4;j++){
        int top2 = top + j;
        imgOrg = &reconsFrame[top2 * codingOption.FrameWidth + left];
        for(int i=0;i<4;i++){
            *imgOrg++ = iClip1(MAX_IMGPEL_VALUE,*block++ + *imgPred++);
        }
    }
}

void decode_one_block(unsigned char *reconsFrame, int block[BLOCKSIZE44], int X, int Y, unsigned char *BlockModeFlag){

    unsigned char cur_pred[BLOCKSIZE44]; 

    intra4x4_dc_pred(reconsFrame, X, Y, cur_pred, BlockModeFlag);

    inverse44_intra(block);

    SampleRecon(reconsFrame, block, cur_pred, X, Y);
}


void intraRecon(unsigned char *reconsFrame, EncodedFrame *encodedFrame){  
    double Qstep = pow( 2.0,(((double)codingOption.IntraQP - 4.0) / 6.0));

    for(int i=0;i<Block44Height;i++){
		int block88Row = (i>>1) * Block88Width;
        for(int j=0;j<Block44Width;j++){
            if(encodedFrame->BlockModeFlag[block88Row + (j>>1)] == INTRA_B){
                int block[BLOCKSIZE44];
				//printf("\nblock(%d %d)\n",i,j);
                //1. CAVLC decoding, de-quant
                read_one_block(block, encodedFrame, j, i, Qstep);
                //2. IDCT, intra predictin
                decode_one_block(reconsFrame, block, j, i, encodedFrame->BlockModeFlag);
            }
        }
    }

}

void copyIntraBlkToSideInfo(unsigned char *sideInfoFrame, unsigned char *reconsFrame, unsigned char *BlockModeFlag){
	int blk88Idx = 0;
	for(int i=0; i<Block88Height; i++){
		int top = i << 3;
		for(int j=0; j<Block88Width; j++){
			//for each block
			int left = j << 3;

			if(BlockModeFlag[blk88Idx++] == INTRA_B){
				for(int m=0; m<8; m++){
					int rowInFrame = (top+m) * codingOption.FrameWidth + left;				
					for(int n=0; n<8; n++){
						sideInfoFrame[rowInFrame + n] = reconsFrame[rowInFrame + n];
					}
				}
			}
		}
	}
}
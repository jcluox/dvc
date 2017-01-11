#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "intra.h"
#include "vlc.h"
#include "inlineFunc.h"
#include "transform.h"


int bitdepth_luma = 8;

#define BLOCK_SHIFT 2
#define CAVLC_LEVEL_LIMIT 2063


//  Notation for comments regarding prediction and predictors.
//  The pels of the 4x4 block are labelled a..p. The predictor pels above
//  are labelled A..H, from the left I..P, and from above left X, as follows:
// 
//   X A B C D E F G H
//   I a b c d
//   J e f g h
//   K i j k l
//   L m n o p
// 

// Predictor array index definitions
#define P_X (PredPel[0])
#define P_A (PredPel[1])
#define P_B (PredPel[2])
#define P_C (PredPel[3])
#define P_D (PredPel[4])
#define P_E (PredPel[5])
#define P_F (PredPel[6])
#define P_G (PredPel[7])
#define P_H (PredPel[8])
#define P_I (PredPel[9])
#define P_J (PredPel[10])
#define P_K (PredPel[11])
#define P_L (PredPel[12])


const int quant_coef[6][BLOCKSIZE44] = {
  {13107, 8066,13107, 8066, 8066, 5243, 8066, 5243, 13107, 8066,13107, 8066, 8066, 5243, 8066, 5243},
  {11916, 7490,11916, 7490, 7490, 4660, 7490, 4660, 11916, 7490,11916, 7490, 7490, 4660, 7490, 4660},
  {10082, 6554,10082, 6554, 6554, 4194, 6554, 4194, 10082, 6554,10082, 6554, 6554, 4194, 6554, 4194},
  { 9362, 5825, 9362, 5825, 5825, 3647, 5825, 3647, 9362, 5825, 9362, 5825, 5825, 3647, 5825, 3647},
  { 8192, 5243, 8192, 5243, 5243, 3355, 5243, 3355, 8192, 5243, 8192, 5243, 5243, 3355, 5243, 3355},
  { 7282, 4559, 7282, 4559, 4559, 2893, 4559, 2893, 7282, 4559, 7282, 4559, 4559, 2893, 4559, 2893}
};

const int dequant_coef[6][BLOCKSIZE44] = {
  {10, 13, 10, 13, 13, 16, 13, 16, 10, 13, 10, 13, 13, 16, 13, 16},
  {11, 14, 11, 14, 14, 18, 14, 18, 11, 14, 11, 14, 14, 18, 14, 18},
  {13, 16, 13, 16, 16, 20, 16, 20, 13, 16, 13, 16, 16, 20, 16, 20},
  {14, 18, 14, 18, 18, 23, 18, 23, 14, 18, 14, 18, 18, 23, 18, 23},
  {16, 20, 16, 20, 20, 25, 20, 25, 16, 20, 16, 20, 20, 25, 20, 25},
  {18, 23, 18, 23, 23, 29, 23, 29, 18, 23, 18, 23, 23, 29, 23, 29}
};

void reset_macroblock(BlockData *currB){

    currB->bits.mb_mode = 0;
    currB->bits.mb_inter = 0;
    currB->bits.mb_cbp = 0;
    currB->bits.mb_y_coeff = 0;
    currB->bits.mb_uv_coeff = 0;
    currB->bits.mb_cb_coeff = 0;
    currB->bits.mb_cr_coeff = 0;
    currB->bits.mb_delta_quant = 0;
    currB->bits.mb_stuffing = 0;


    currB->mode      = 0; // WZ
    currB->nonzero   = 0;
}

void printblock(int *block){
    for(int i=0;i<4;i++){
        for(int j=0;j<4;j++){
            printf("%3d ",block[(i<<2)+j]);
        }
        printf("\n");
    }
}


void forward44_intra(int *blk){
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

void getNeighbour(int X,int Y,int refB,int xN, int yN, PixelPos *pix, unsigned char *intraBlockFlag){

    switch(refB){
        case 0:
            // A
            if(xN < 0){
                pix->available = 0;
            }else{
                if(intraBlockFlag[(Y>>1)*Block88Width + ((X-1)>>1)] == 1){
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
				if(intraBlockFlag[((Y-1)>>1)*Block88Width + (X>>1)] == 1){
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


void block_intra44(unsigned char *img_enc, int X, int Y, unsigned char *cur_pred, EncodedFrame *encodedFrame){

    unsigned int dc_pred_value = 1<<(bitdepth_luma - 1);
    int block_available_up;
    int block_available_left;
    //    int block_available_up_left;
    //    int block_available_up_right;
    //unsigned char *img_enc = reconsFrames[frameIdx];
    unsigned char *img_pel;
    //unsigned char cur_pred[BLOCKSIZE44];
    PixelPos pix_a[4],pix_b;
    //PixelPos pix_c,pix_d;
    unsigned char  PredPel[13];



    int ioff = (X<<2);
    int joff = (Y<<2);


    for(int i=0;i<4;i++){
        getNeighbour(X,Y,0,ioff - 1,joff + i,&pix_a[i],encodedFrame->intraBlockFlag);	// left
    }
    getNeighbour(X,Y,1,ioff,joff - 1,&pix_b,encodedFrame->intraBlockFlag);	// up
    /*
       getNeighbour(X,Y,2,ioff + 4,joff - 1,&pix_c);
       getNeighbour(X,Y,3,ioff - 1,joff - 1,&pix_d);
       */

    block_available_left     = pix_a[0].available;
    block_available_up       = pix_b.available;
    //    block_available_up_right = pix_c.available;
    //    block_available_up_left  = pix_d.available;

    // form predictor pels
    if (block_available_up)
    {
		
        img_pel = &img_enc[(pix_b.pos_y)*codingOption.FrameWidth + pix_b.pos_x];
        P_A = *(img_pel++);
        P_B = *(img_pel++);
        P_C = *(img_pel++);
        P_D = *(img_pel);                

    }
    else
    {
        P_A = P_B = P_C = P_D = dc_pred_value;
    }
    /*    
          if (block_available_up_right)
          {
          img_pel = &img_enc[pix_c.pos_y * codingOption.FrameWidth + pix_c.pos_x];
          P_E = *(img_pel++);
          P_F = *(img_pel++);
          P_G = *(img_pel++);
          P_H = *(img_pel);
          }
          else
          {
          P_E = P_F = P_G = P_H = P_D;
          }
          */
    if (block_available_left)
    {
        P_I = img_enc[pix_a[0].pos_y*codingOption.FrameWidth + pix_a[0].pos_x];
        P_J = img_enc[pix_a[1].pos_y*codingOption.FrameWidth + pix_a[1].pos_x];
        P_K = img_enc[pix_a[2].pos_y*codingOption.FrameWidth + pix_a[2].pos_x];
        P_L = img_enc[pix_a[3].pos_y*codingOption.FrameWidth + pix_a[3].pos_x];

    }
    else
    {
        P_I = P_J = P_K = P_L = dc_pred_value;
    }
    /*
       if (block_available_up_left)
       {
       P_X = img_enc[pix_d.pos_y*codingOption.FrameWidth + pix_d.pos_x];
       }
       else
       {
       P_X = dc_pred_value;
       }
       */
    // DC prediction
    int s0 = 0;
    if (block_available_up && block_available_left)
    {
        // no edge
        s0 = (P_A + P_B + P_C + P_D + P_I + P_J + P_K + P_L + 4) >> (BLOCK_SHIFT + 1);
    }
    else if (!block_available_up && block_available_left)
    {
        // upper edge
        s0 = (P_I + P_J + P_K + P_L + 2) >> BLOCK_SHIFT;
    }
    else if (block_available_up && !block_available_left)
    {
        // left edge
        s0 = (P_A + P_B + P_C + P_D + 2) >> BLOCK_SHIFT;
    }
    else //if (!block_available_up && !block_available_left)
    {
        // top left corner, nothing to predict from
        s0 = dc_pred_value;
    }

    // store DC prediction, and find residual
    for (int j=0; j < BLOCKSIZE44; j++){
        *cur_pred++ = (unsigned char) s0;        
    }


}


void generate_pred_error(int frameIdx,int X,int Y,int *block,unsigned char *cur_pred){
    int top = Y << 2;
    int left = X << 2;    
    unsigned char *img_ori = oriFrames[frameIdx];
    int *block_line;
    unsigned char *cur_line,*prd_line;
    block_line = block;
    prd_line = cur_pred;

    for (int j=0; j < 4; j++){
        int top2 = top + j;
        //block_line = &block[(j<<2)];
        cur_line = &img_ori[top2 * codingOption.FrameWidth + left];
        //prd_line = &cur_pred[(j<<2)];
        for (int i=0; i < 4; i++){
            //int left2 = left + i;
            //int pix_idx = top2 * codingOption.FrameWidth + left2;
            
            *block_line++ = (int)(*cur_line++ - *prd_line++);            
        }
    }
}


int quant_4x4(int *block, int *qblock,int qbits,int offset,BlockData *currblock,double Qstep,const int *levelscale,const int *invlevelscale){
    int nonzero = false;
    int scaled_coeff,level,run=0;
    static int *m7;
    int *ACL = currblock->level;
    int *ACR = currblock->run;
    int qp_per = qp_per_matrix[codingOption.IntraQP];
    int q_bits = Q_BITS + qp_per;
    

    for(int i=0;i<BLOCKSIZE44;i++){
        
        m7 = &block[zigzag[i]];

        if(*m7 != 0){
            //scaled_coeff = iabs(*m7) * levelscale[zigzag[i]];
            scaled_coeff = iabs(*m7) * MF[zigzag[i]];
            //printf("\nvalue = %d\n",*m7);
            //printf("A: %d vs %d\n", iabs(*m7)*MF[zigzag[i]],iabs(*m7)*levelscale[zigzag[i]]);
            level = (scaled_coeff + offset) >> q_bits;

            qblock[zigzag[i]] = isignab(level, *m7);

            if(level != 0){
                level = imin(level, CAVLC_LEVEL_LIMIT);

                level = isignab(level, *m7);
                // rescalling
                *m7 = ROUND((double)level * Qstep * preScale[zigzag[i]]);
                //*m7 = rshift_rnd_sf(((level * (invlevelscale[zigzag[i]] << 4)) << qp_per), 4);
                //printf("B: %d vs %d\n", ROUND((double)level *Qstep * preScale[zigzag[i]]),rshift_rnd_sf(((level * invlevelscale[zigzag[i]]) << qp_per), 4));
                *ACL++ = level;
                *ACR++ = run;


                run = 0;
                
                nonzero = true;

            }else{
                run++;
                (*m7) = 0;
                //*ACL++ = 0;
            }
        }else{
            qblock[zigzag[i]] = 0;
            run++;
        }
    }
    *ACL = 0;
    //printblock(qblock);

    return nonzero;
}


void SampleRecon(int *block,unsigned char *cur_pred,int frameIdx,int X,int Y){
    unsigned char *reconsFrame = (unsigned char *)reconsFrames[frameIdx];
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

int dct_44(int *block,int frameIdx,int X, int Y,int qbits,int offset,EncodedFrame *encodedFrame,double Qstep,unsigned char *cur_pred,int block44pos){
    unsigned char *reconsFrame = reconsFrames[frameIdx];
    int nonzero = false;
    int qblock[BLOCKSIZE44];
    BlockData *currblock = &(encodedFrame->blockInfo[block44pos]);    
    int qp_rem = qp_rem_matrix[codingOption.IntraQP];
    const int *levelscale = quant_coef[qp_rem]; 
    const int *invlevelscale = dequant_coef[qp_rem];

    forward44_intra(block);
    
    nonzero = quant_4x4(block,qblock,qbits,offset,currblock,Qstep,levelscale,invlevelscale); 
    
    int count = 0;
    for(int i=0;i<BLOCKSIZE44;i++){
        if(qblock[i] == 0){
            count++;
        }
    }
    
    double rho = (double)count/16.0;
    encodedFrame->Z[block44pos] = 1.0-rho;
    encodedFrame->E[block44pos] = abs(qblock[0]);
    
    
    if(nonzero){
        // Inverse transform     
        inverse44_intra(block);
		/*printf("residual:\n");
		for(int j=0;j<4;j++){
           
            for(int i=0;i<4;i++){
                printf("%3d ",block[j*4+i]);
            }
            printf("\n");
        }*/

        // generate final block
        SampleRecon(block,cur_pred,frameIdx,X,Y);
		
		
    }else{
        // if (nonzero) => No transformed residual. Just use prediction.
        int top = Y << 2;
        int left = X << 2;    
        for(int i=0;i<4;i++){
            int top2 = top + i;
            memcpy(&reconsFrame[top2 * codingOption.FrameWidth + left], &cur_pred[(i<<2)], sizeof(unsigned char)*4);
        }
    }

    return nonzero;
}


int intra_prediction(int frameIdx,int X,int Y,EncodedFrame *encodedFrame,int block44pos,int qbits,int offset,double Qstep,int write_flag,int recal_intra)
{
    int block[BLOCKSIZE44];
    int nonzero;
    unsigned char cur_pred[BLOCKSIZE44];
    int rate=0;

    if(write_flag==0){

        // calculate prediction
        block_intra44(reconsFrames[frameIdx],X,Y,cur_pred,encodedFrame);


        // calculate residual
        generate_pred_error(frameIdx,X,Y,block,cur_pred);


        // DCT (transform, quantization, de-quantization, inv-transform)
        nonzero = dct_44(block,frameIdx,X,Y,qbits,offset,encodedFrame,Qstep,cur_pred,block44pos);
        // get distortion (SSD) of 4x4 block
        // now I ignore it!


        if(recal_intra){
            rate = writeCoeff4x4_CAVLC(encodedFrame,X,Y,block44pos,write_flag);   
            encodedFrame->Rspent[block44pos] = rate;
        }else{
            rate = 0;
        }
    }else{
        rate = writeCoeff4x4_CAVLC(encodedFrame,X,Y,block44pos,write_flag);
        //encodedFrame->Rspent[block44pos] = rate; 
    }
    //printf("rate = %d\n",rate);
    return rate;
}


void real_intra(EncodedFrame *encodedFrame, int frameIdx){
    int realrate = 0;
    int qp_per = qp_per_matrix[codingOption.IntraQP];
    int qbits = Q_BITS + qp_per;
    //int qbits = (int)(15.0 + floor((double)codingOption.IntraQP / 6.0));
    int offset = (int)( ((double)(1<<qbits) /3.0) + 0.5);
    double Qstep = pow( 2.0,(((double)codingOption.IntraQP - 4.0) / 6.0));
    //int write_flag=1;
    unsigned char *recon = reconsFrames[frameIdx];

	int block44pos = 0;
    for(int i=0;i<Block44Height;i++){
		int block88Row = (i>>1) * Block88Width;
        for(int j=0;j<Block44Width;j++){
            if(encodedFrame->intraBlockFlag[block88Row + (j>>1)] == 1){
                //int temprate=0;
                //printf("\nblock[%2d,%2d] (%d):\n",i,j,block44pos);
                realrate += intra_prediction(frameIdx,j,i,encodedFrame,block44pos,qbits,offset,Qstep,1,1);
                //realrate+=temprate;
                //printf("Real Bit Rate = %d\n",temprate);
            }
			block44pos++;
        }
    }
    /*
       if(encodedFrame->skipMode == 1){
       printf(" , Intra Real Rate = %d\n",realrate);
       }else{
       printf("-, Intra Real Rate = %d\n",realrate);
       }
       */
}


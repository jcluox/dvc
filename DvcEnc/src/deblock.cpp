#include <math.h>
#include <string.h>
#include "global.h"
#include "inlineFunc.h"
#include "error.h"
#include "quantization.h"


#define BSMAX 0.14

#define WZ          0
#define INTRA       1
#define SKIP        2

#define NO_F        0
#define WZ_mode     1
#define ISWZ_mode   2

static const unsigned char ALPHA_TABLE[52]  = {0,0,0,0,0,0,0,0,0,0,0,0, 
                                      0,0,0,0,4,4,5,6,  
                                      7,8,9,10,12,13,15,17,  
                                      20,22,25,28,32,36,40,45,  
                                      50,56,63,71,80,90,101,113,  
                                      127,144,162,182,203,226,255,255} ;
static const unsigned char  BETA_TABLE[52]  = {0,0,0,0,0,0,0,0,0,0,0,0, 
                                      0,0,0,0,2,2,2,3,  
                                      3,3,3, 4, 4, 4, 6, 6,   
                                      7, 7, 8, 8, 9, 9,10,10,  
                                      11,11,12,12,13,13, 14, 14,   
                                      15, 15, 16, 16, 17, 17, 18, 18} ;

static const unsigned char CLIP_TAB[52][5]  =
{
    { 0, 0, 0, 0, 0},{ 0, 0, 0, 0, 0},{ 0, 0, 0, 0, 0},{ 0, 0, 0, 0, 0},{ 0, 0, 0, 0, 0},{ 0, 0, 0, 0, 0},{ 0, 0, 0, 0, 0},{ 0, 0, 0, 0, 0},
    { 0, 0, 0, 0, 0},{ 0, 0, 0, 0, 0},{ 0, 0, 0, 0, 0},{ 0, 0, 0, 0, 0},{ 0, 0, 0, 0, 0},{ 0, 0, 0, 0, 0},{ 0, 0, 0, 0, 0},{ 0, 0, 0, 0, 0},
    { 0, 0, 0, 0, 0},{ 0, 0, 0, 1, 1},{ 0, 0, 0, 1, 1},{ 0, 0, 0, 1, 1},{ 0, 0, 0, 1, 1},{ 0, 0, 1, 1, 1},{ 0, 0, 1, 1, 1},{ 0, 1, 1, 1, 1},
    { 0, 1, 1, 1, 1},{ 0, 1, 1, 1, 1},{ 0, 1, 1, 1, 1},{ 0, 1, 1, 2, 2},{ 0, 1, 1, 2, 2},{ 0, 1, 1, 2, 2},{ 0, 1, 1, 2, 2},{ 0, 1, 2, 3, 3},
    { 0, 1, 2, 3, 3},{ 0, 2, 2, 3, 3},{ 0, 2, 2, 4, 4},{ 0, 2, 3, 4, 4},{ 0, 2, 3, 4, 4},{ 0, 3, 3, 5, 5},{ 0, 3, 4, 6, 6},{ 0, 3, 4, 6, 6},
    { 0, 4, 5, 7, 7},{ 0, 4, 5, 8, 8},{ 0, 4, 6, 9, 9},{ 0, 5, 7,10,10},{ 0, 6, 8,11,11},{ 0, 6, 8,13,13},{ 0, 7,10,14,14},{ 0, 8,11,16,16},
    { 0, 9,12,18,18},{ 0,10,13,20,20},{ 0,11,15,23,23},{ 0,13,17,25,25}
} ;

void deblock_filter(int dir, int mode, double Bs, unsigned char *img, int block1_pos_x,int block1_pos_y, int block2_pos_x, int block2_pos_y){
    
    static unsigned char   L2, L1, L0, R0, R1, R2, L3, R3;
    static int      C0, tc0, dif, RL0;
    static int      ap, aq, Strng;
    static int      QP = codingOption.IntraQP;
    static int      Alpha, Beta, small_gap;
    static int      indexA, indexB;
    static const unsigned char* ClipTab;
    static int inc_dim, inc_dim3;
    int  bitdepth_scale = 1;
    unsigned char *blockP, *blockQ;
    int max_imgpel_value = MAX_IMGPEL_VALUE;
    
    
    inc_dim = dir ? codingOption.FrameWidth : 1;
    inc_dim3 = inc_dim*3;


    indexA = QP;
    indexB = QP;

    Alpha   = ALPHA_TABLE[indexA] * bitdepth_scale;
    Beta    = BETA_TABLE [indexB] * bitdepth_scale;
    ClipTab = CLIP_TAB[indexA];
    
    for(int line = 0;line<4;line++){
        int yd = dir ? 0 : line;
        int xd = dir ? line : 0;

        int idx1 = ((block1_pos_y<<2) + yd) * codingOption.FrameWidth + (block1_pos_x<<2) + xd;
        int idx2 = ((block2_pos_y<<2) + yd) * codingOption.FrameWidth + (block2_pos_x<<2) + xd;
        blockP = &img[idx1];
        blockQ = &img[idx2];
        
        L3 = *blockP;
        L2 = *(blockP += inc_dim);
        L1 = *(blockP += inc_dim);
        L0 = *(blockP += inc_dim);
        R0 = *blockQ;
        R1 = *(blockQ += inc_dim);
        R2 = *(blockQ += inc_dim);
        R3 = *(blockQ += inc_dim);
        
        //printf("%3d %3d %3d %3d \t %3d %3d %3d %3d",L3,L2,L1,L0,R0,R1,R2,R3);
        
        if(iabs(R0 - L0) < Alpha){
            if((iabs(R0 - R1) < Beta) && (iabs(L0 - L1) < Beta)){

                if(mode == ISWZ_mode){
                    
                    Strng = (int)Bs;
                    if(Strng == 4){  // Intra Strong Filter
                        
                        
                        RL0 = L0 + R0;
                        small_gap = (iabs( R0 - L0 ) < ((Alpha >> 2) + 2));
                        aq  = ( iabs( R0 - R2) < Beta ) & small_gap;
                        ap  = ( iabs( L0 - L2) < Beta ) & small_gap;
                        //printf(" => Strong");
                        if (ap){
                            *blockP               = (unsigned char)  (( R1 + ((L1 + RL0) << 1) +  L2 + 4) >> 3);
                            *(blockP -= inc_dim)  = (unsigned char)  (( L2 + L1 + RL0 + 2) >> 2);
                            *(blockP -  inc_dim)  = (unsigned char) ((((L3 + L2) <<1) + L2 + L1 + RL0 + 4) >> 3);                
                        }else{
                            *blockP               = (unsigned char) (((L1 << 1) + L0 + R1 + 2) >> 2) ;                
                        }

                        if (aq){
                            *(blockQ -= inc_dim3) = (unsigned char) (( L1 + ((R1 + RL0) << 1) +  R2 + 4) >> 3);
                            *(blockQ += inc_dim ) = (unsigned char) (( R2 + R0 + L0 + R1 + 2) >> 2);
                            *(blockQ +  inc_dim ) = (unsigned char) ((((R3 + R2) <<1) + R2 + R1 + RL0 + 4) >> 3);
                        }else{
                            *(blockQ -  inc_dim3) = (unsigned char) (((R1 << 1) + R0 + L1 + 2) >> 2);
                        }

                    }else{          // Default Filter

                        RL0 = (L0 + R0 + 1) >> 1;
                        aq  = (iabs(R0 - R2) < Beta);
                        ap  = (iabs(L0 - L2) < Beta);

                        C0  = ClipTab[ Strng ] * bitdepth_scale;
                        tc0  = (C0 + ap + aq);
                        dif = iClip3( -tc0, tc0, (((R0 - L0) << 2) + (L1 - R1) + 4) >> 3 );
                        
                        //printf(" => C0 = %d, dif = %d",C0,dif);

                        if( ap ){
                            *(blockP - inc_dim) += iClip3( -C0,  C0, (L2 + RL0 - (L1<<1)) >> 1 );
                        }

                        *blockP                 = (unsigned char) iClip1(max_imgpel_value, L0 + dif);
                        *(blockQ -= inc_dim3)   = (unsigned char) iClip1(max_imgpel_value, R0 - dif);

                        if( aq ){
                            *(blockQ + inc_dim) += iClip3( -C0,  C0, (R2 + RL0 - (R1<<1)) >> 1 );
                        }

                    }
                
                }else if(WZ_mode){
                    double mu = BSMAX * 0.5;
                    if(Bs > mu){    // Strong Filter
                        
                        RL0 = L0 + R0;
                        small_gap = (iabs( R0 - L0 ) < ((Alpha >> 2) + 2));
                        aq  = ( iabs( R0 - R2) < Beta ) & small_gap;
                        ap  = ( iabs( L0 - L2) < Beta ) & small_gap;
                        //printf(" => Strong");
                        if (ap){
                            *blockP               = (unsigned char)  (( R1 + ((L1 + RL0) << 1) +  L2 + 4) >> 3);
                            *(blockP -= inc_dim)  = (unsigned char)  (( L2 + L1 + RL0 + 2) >> 2);
                            *(blockP -  inc_dim)  = (unsigned char) ((((L3 + L2) <<1) + L2 + L1 + RL0 + 4) >> 3);                
                        }else{
                            *blockP               = (unsigned char) (((L1 << 1) + L0 + R1 + 2) >> 2) ;                
                        }

                        if (aq){
                            *(blockQ -= inc_dim3) = (unsigned char) (( L1 + ((R1 + RL0) << 1) +  R2 + 4) >> 3);
                            *(blockQ += inc_dim ) = (unsigned char) (( R2 + R0 + L0 + R1 + 2) >> 2);
                            *(blockQ +  inc_dim ) = (unsigned char) ((((R3 + R2) <<1) + R2 + R1 + RL0 + 4) >> 3);
                        }else{
                            *(blockQ -  inc_dim3) = (unsigned char) (((R1 << 1) + R0 + L1 + 2) >> 2);
                        }


                    }else if(Bs <= mu && Bs > 0.0){ // Default Filter

                        RL0 = (L0 + R0 + 1) >> 1;
                        aq  = (iabs(R0 - R2) < Beta);
                        ap  = (iabs(L0 - L2) < Beta);

                        C0  = (int)(Bs * (pow(2,((double)QP / 6.0)) - 1));
                        tc0  = (C0 + ap + aq);
                        dif = iClip3( -tc0, tc0, (((R0 - L0) << 2) + (L1 - R1) + 4) >> 3 );
                        
                        //printf(" => C0 = %d, dif = %d",C0,dif);

                        if( ap ){
                            *(blockP - inc_dim) += iClip3( -C0,  C0, (L2 + RL0 - (L1<<1)) >> 1 );
                        }

                        *blockP                 = (unsigned char) iClip1(max_imgpel_value, L0 + dif);
                        *(blockQ -= inc_dim3)   = (unsigned char) iClip1(max_imgpel_value, R0 - dif);

                        if( aq ){
                            *(blockQ + inc_dim) += iClip3( -C0,  C0, (R2 + RL0 - (R1<<1)) >> 1 );
                        }
                    }else{
                        // no filter 
                    }
                }else{
                    errorMsg("De-blocking Mode");
                }
            }

        }

        //printf("\n");

    }
    /*
    printf("After de-blocking:\n");
    for(int line = 0;line<4;line++){

        int yd = dir ? 0 : line;
        int xd = dir ? line : 0;

        int idx1 = ((block1_pos_y<<2) + yd) * codingOption.FrameWidth + (block1_pos_x<<2) + xd;
        int idx2 = ((block2_pos_y<<2) + yd) * codingOption.FrameWidth + (block2_pos_x<<2) + xd;
        blockP = &img[idx1];
        blockQ = &img[idx2];
        
        L3 = *blockP;
        L2 = *(blockP += inc_dim);
        L1 = *(blockP += inc_dim);
        L0 = *(blockP += inc_dim);
        R0 = *blockQ;
        R1 = *(blockQ += inc_dim);
        R2 = *(blockQ += inc_dim);
        R3 = *(blockQ += inc_dim);
        
        printf("%3d %3d %3d %3d \t %3d %3d %3d %3d\n",L3,L2,L1,L0,R0,R1,R2,R3);
    }*/

}

int deblock_44_decision(
        unsigned char *reconimg,
        int *trans_sideImg,
        int *trans_reconImg,
        unsigned char *type_map,
        int block1_pos_x,
        int block1_pos_y, 
        int block2_pos_x, 
        int block2_pos_y,
        double *Bs,
        double weight[BLOCKSIZE44],
        int begin)
{

    int mode = 0;
    //printf("block (%d, %d) with block (%d, %d)\n",block1_pos_y,block1_pos_x,block2_pos_y,block2_pos_x);
    //int pos1 = block1_pos_y * Block44Width + block1_pos_x;
    //int pos2 = block2_pos_y * Block44Width + block2_pos_x;
	int pos1 = (block1_pos_y>>1) * Block88Width + (block1_pos_x>>1);
    int pos2 = (block2_pos_y>>1) * Block88Width + (block2_pos_x>>1);

    if(type_map[pos1] == INTRA && type_map[pos2] == INTRA){        
        mode = ISWZ_mode;
        *Bs = 3.0;
    }else if(type_map[pos1] == INTRA || type_map[pos2] == INTRA){
        mode = ISWZ_mode;
        if(type_map[pos1] == WZ || type_map[pos2] == WZ){
            *Bs = 4.0;
        }else{            
            *Bs = 3.0;
        }
    }else if(type_map[pos1] == SKIP && type_map[pos2] == SKIP){
        mode = NO_F;
        *Bs = 0.0;
    }else if(type_map[pos1] == SKIP || type_map[pos2] == SKIP){ // not sure        
        mode = ISWZ_mode;
        if(encodedFrames[begin].isWZ == 1){
            if(type_map[pos1] == SKIP){
                if(encodedFrames[begin].intraBlockFlag[pos1] == 1)
                    *Bs = 2.0;
                else
                    *Bs = 1.0;
            }else{
                //if(bitmap_all[begin][pos2] == INTRA)
				if(encodedFrames[begin].intraBlockFlag[pos2] == 1)
                    *Bs = 2.0;
                else
                    *Bs = 1.0;
            }
        }else{
            *Bs = 2.0;
        }
    }else if(type_map[pos1] == WZ && type_map[pos2] == WZ){

        double phi1 = 0.0;
        int sum1 = 0;
        int sq_sum1 = 0;
        for(int m=0;m<4;m++){
            int idx = ((block1_pos_y<<2) + m) * codingOption.FrameWidth + (block1_pos_x<<2);
            for(int n=0;n<4;n++){
                if((trans_reconImg[idx + n] ^ trans_sideImg[idx + n]) != 0 ){
                    phi1 += weight[(m<<2)+n];
                }
                sum1 += reconimg[idx + n];
                sq_sum1 += reconimg[idx + n] * reconimg[idx + n];
            }
        }

        double var1 = (double)sq_sum1 / 16.0 - ((double)sum1 / 16.0) * ((double)sum1 / 16.0);

        double chi1 = 0.0;
        if(var1 < 50.0){
            chi1 = 1.0;
        }else{
            chi1 = 0.0;
        }
        double phi2 = 0.0;
        int sum2 = 0;
        int sq_sum2 = 0;
        for(int m=0;m<4;m++){
            int idx = ((block2_pos_y<<2) + m) * codingOption.FrameWidth + (block2_pos_x<<2);
            for(int n=0;n<4;n++){
                if((trans_reconImg[idx + n] ^ trans_sideImg[idx + n]) != 0){
                    phi2 += weight[(m<<2) + n];
                }
                sum2 += reconimg[idx + n];
                sq_sum2 += reconimg[idx + n] * reconimg[idx + n];
            }
        }


        double var2 = (double)sq_sum2 / 16.0 - ((double)sum2 / 16.0) * ((double)sum2 / 16.0);

        double chi2 = 0.0;
        if(var2 < 50){
            chi2 = 1.0;
        }else{
            chi2 = 0.0;
        }

        //printf("phi2 = %.2lf, chi2 = %.1lf\n",phi2,chi2);


        *Bs = 0.0;
        //printf("phi2 = %.2lf, chi2 = %.1lf\n",phi2,chi2);
        double P = 0.9;
        //*Bs = BSMAX * ( (chi1+chi2) / 2.0 );
        *Bs = BSMAX * (P*( (phi1+phi2) / 2.0 ) + (1.0 - P)*( (chi1+chi2) / 2.0 ));
        //printf("Bs = %lf\n",*Bs);

        if(*Bs > 0.0){
            mode = WZ_mode;
        }else{
            mode = NO_F;
        }
    }else{
        mode = NO_F;
    }
    return mode;
}

void deblock(unsigned char *deblockimg, // deblocking frames
        unsigned char *recon,      // pixel-domain recon frame            
        int *trans_sideImg,
        int *trans_reconImg,
        unsigned char *type_map,
        int begin)
{

    double Bs = 0.0;
    double weight[BLOCKSIZE44];

    memcpy(deblockimg, recon, sizeof(unsigned char) * FrameSize);
    int sum = 0;
    for(int i=0;i<BLOCKSIZE44;i++)
        sum += Qtable[i];
    for(int i=0;i<BLOCKSIZE44;i++)
        weight[i] = (double)Qtable[i] / (double)sum;

    for(int dir=0; dir < 2; dir++){
        int xd = dir ? 0:1;
        int yd = dir ? 1:0;

        for(int i=0;i<(Block44Height-yd);i++){
            for(int j=0;j<(Block44Width-xd);j++){
                // decision
                int mode = deblock_44_decision(recon,trans_sideImg,trans_reconImg,type_map,j,i,j+xd,i+yd,weight,&Bs,begin);

                //printf("Bs = %lf\n",Bs);

                // filtering
                if(mode == WZ_mode || mode == ISWZ_mode){ 
                    deblock_filter(dir, mode, Bs, deblockimg, j, i, j+xd, i+yd);
                }
                //printf("\n");
            }
        }
    }

}

void init_type_map(unsigned char *type_map, unsigned char *intraBlockFlag, unsigned char *skipBlockFlag){
	for(int i=0; i<Block88Num; i++){
		if(skipBlockFlag[i] == 1)
			type_map[i] = 2;	// skip
		else if(intraBlockFlag[i] == 1)
			type_map[i] = 1;	// Intra
		else
			type_map[i] = 0;	// WZ
	}
}

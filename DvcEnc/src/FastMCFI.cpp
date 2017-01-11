#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "FastMCFI.h"
#include "error.h"

int FS_P[8][2] = {
    {1,0},
    {-1,0},
    {0,1},
    {0,-1},
    {2,0},
    {-2,0},
    {0,2},
    {0,-2}
};

void AverInter(unsigned char *sideInfoFrame, int FrameIdx, int begin, int end, int *dist){    
    double interFC=(double)(end - FrameIdx) / (double)(end - begin);
    double interPC=(double)(FrameIdx - begin) / (double)(end - begin);
    // Average Interpolation and just copy
    for(int i=0;i<codingOption.FrameHeight;i++){
        for(int j=0;j<codingOption.FrameWidth;j++){
            int index = i*codingOption.FrameWidth+j;
            //sideInfoFrame[index] = (SIFrames[begin][index] + SIFrames[end][index] + 1) >> 1;
            
            sideInfoFrame[index] = (unsigned char)((double)reconsFrames[begin][index]*interPC + (double)reconsFrames[end][index]*interFC + 0.5);
            //printf("%d %d %d\n",sideInfoFrame[index],SIFrames[begin][index],SIFrames[end][index]);
        }         
    }
    // distortion calculation
    for(int i=0;i<Block88Height;i++){
        int top = i<< 3;
        for(int j=0;j<Block88Width;j++){
            int left = j << 3;
            int block_idx = i*Block88Width+j;
            int dist1=0;
            //int dist2=0;
            //int dist3=0;
            

            for(int ii=0;ii<8;ii++){
                int y = (top + ii)*codingOption.FrameWidth + left;
                for(int jj=0;jj<8;jj++){
                    int idx = y + jj;
                    
                    dist1 += (int)abs(reconsFrames[begin][idx]-reconsFrames[end][idx]);
                    //dist1 += (int)abs(sideInfoFrame[idx]-oriFrames[FrameIdx][idx]);
                    //dist2 += (int)abs(SIFrames[begin][idx]-oriFrames[FrameIdx][idx]);
                    //dist3 += (int)abs(SIFrames[end][idx]-oriFrames[FrameIdx][idx]);
                }
            }
            dist[block_idx] = dist1;

            /*
            if(dist1 <= dist2 && dist1 <= dist3){
                dist[block_idx] = dist1;   
            }else if(dist2 < dist1 && dist2 <= dist3){
                dist[block_idx] = dist2;
                for(int ii=0;ii<8;ii++){
                    int y = (top + ii) * codingOption.FrameWidth + left;
                    for(int jj=0;jj<8;jj++){
                        int idx = y + jj;
                        sideInfoFrame[idx] = SIFrames[begin][idx];
                    }
                }
            }else if(dist3 < dist1 && dist3 < dist2){
                dist[block_idx] = dist3;
                for(int ii=0;ii<8;ii++){
                    int y = (top + ii) * codingOption.FrameWidth + left;
                    for(int jj=0;jj<8;jj++){
                        int idx = y + jj;
                        sideInfoFrame[idx] = SIFrames[end][idx];
                    }
                }

            }else{
                errorMsg("AI distortion calculation");
            }*/


            //printf("min dist = %d\n",dist[block_idx]);
        }
    }

}

int position(int pos, int lower, int upper){
    if(pos < lower){
        return lower;
    }
	else if(pos > upper){
        return upper;
    }
	else{
        return pos;
    }
}

static int cmpint(const void *p1, const void *p2){
    return (*((const int *)p1) < *((const int *)p2));

}

int pattern_search(int pn, int ini_mvx, int ini_mvy, int *best_mvx, int *best_mvy, int *min_SAD, int begin, int end, int top, int left){
    int SAD[4]; 
    //double interFC=(double)(end - FrameIdx) / (double)(end - begin);
    //double interPC=(double)(FrameIdx - begin) / (double)(end - begin);
    int revalue = 0;
    int ppn;

    memset(SAD,0,sizeof(int)*4);    

    ppn = (pn % 2) << 2;

    for(int pi=0;pi<4;pi++){
        int mvx = ini_mvx+FS_P[ppn+pi][0];
        int mvy = ini_mvy+FS_P[ppn+pi][1];

        int pattern_skip = 0;
        if(pn == 2){
            for(int m = 0; m<4; m++){
                if(mvx == FS_P[m][0] && mvy == FS_P[m][1]){
                    pattern_skip = 1;
                }                
            }
        }
        if(pattern_skip==0){

            //printf("P[%d] : (%d,%d)\t",pi,mvx,mvy);

            for(int ii=0;ii<8;ii++){
                int y = top + ii;                        
                int py = position((y + mvy),0,codingOption.FrameHeight);
                int fy = position((y - mvy),0,codingOption.FrameHeight);



                for(int jj=0;jj<8;jj++){
                    int x = left + jj;
                    //int idx = y*codingOption.FrameWidth + x;

                    int px = position((x + mvx),0,codingOption.FrameWidth);
                    int fx = position((x - mvx),0,codingOption.FrameWidth);
                    int past_idx = py*codingOption.FrameWidth + px;
                    int future_idx = fy*codingOption.FrameWidth + fx;

                    //int mc = (int)((double)SIFrames[begin][past_idx] * interPC + (double)SIFrames[end][future_idx] * interFC);
                    
                    SAD[pi] += abs(reconsFrames[begin][past_idx] - reconsFrames[end][future_idx]);
                    //SAD[pi] += abs(mc - oriFrames[FrameIdx][idx]);

                }                        
            }
            if(SAD[pi] < (*min_SAD)){
                *best_mvx = mvx;
                *best_mvy = mvy;
                (*min_SAD) = SAD[pi];
                revalue = 1;
            }
            //printf("SAD[%d] = %d\n",pi,SAD[pi]);
        }

    }

    return revalue;

}

void FastSearch(unsigned char *sideInfoFrame, int FrameIdx, int begin,int end, int *dist, EncodedFrame *encodedFrame,double Tper){
    int Fast_num = (int)(Tper * (double)Block88Num);
    int *dist2=(int*)malloc(sizeof(int)*Block88Num);
    int min_dist;    
    double interFC=(double)(end - FrameIdx) / (double)(end - begin);
    double interPC=(double)(FrameIdx - begin) / (double)(end - begin);

    //printf("FC = %lf, PC = %lf\n",interFC,interPC);

    memcpy(dist2,dist,sizeof(int)*Block88Num);


    // sorting of distortion
    qsort((void*)dist2,Block88Num,sizeof(int),cmpint);
    min_dist = dist2[Fast_num-1];

    //printf("Fast Threshold = %d\n",Fast_num);
    //printf("Min distortion = %d\n",min_dist);


    for(int i=0;i<Block88Height;i++){
        int top = i<< 3;
        for(int j=0;j<Block88Width;j++){
            int left = j << 3;
            //printf("top = %d, left = %d\n",top,left);
            int block_idx = i*Block88Width+j;
            int step2_flag = 0;
            int step3_flag = 0;
            
            if(encodedFrame->skipBlockFlag[block_idx] == 1){
                for(int m=0;m<8;m++){
                    int idx = (top + m) * codingOption.FrameWidth + left;
                    for(int n=0;n<8;n++){
                        sideInfoFrame[idx + n] = oriFrames[begin][idx+ n];
                    }
                }
            }else{

                if(dist[block_idx] > min_dist){
                    // Step 1
                    int best_mvx = 0;
                    int best_mvy = 0;
                    int ini_mvx = 0;
                    int ini_mvy = 0;
                    int min_SAD = dist[block_idx];

                    //printf("\ndist[%d] = %d\n",block_idx,dist[block_idx]);

                    //printf("FS Step 1...\n");

                    step2_flag = pattern_search(0,ini_mvx,ini_mvy,&best_mvx,&best_mvy,&min_SAD,begin,end,top,left);


                    //printf("min SAD = %d, best MV(%d,%d)\n",min_SAD,best_mvx,best_mvy);

                    // Step 2
                    if(step2_flag){
                        //printf("FS Step 2...\n");
                        step3_flag = pattern_search(1,ini_mvx,ini_mvy,&best_mvx,&best_mvy,&min_SAD,begin,end,top,left);
                        //printf("min SAD = %d, best MV(%d,%d)\n",min_SAD,best_mvx,best_mvy);
                    }



                    // Step 3
                    if(step3_flag){
                        //printf("FS Step 3...\n");
                        ini_mvx = best_mvx;
                        ini_mvy = best_mvy;
                        pattern_search(2,ini_mvx,ini_mvy,&best_mvx,&best_mvy,&min_SAD,begin,end,top,left);
                        //printf("min SAD = %d, best MV(%d,%d)\n",min_SAD,best_mvx,best_mvy);
                    }

                    dist[block_idx] = min_SAD;

                    // assign pixel value
                    if(step2_flag){

                        for(int ii=0;ii<8;ii++){
                            int y = top + ii;                        
                            int py = position((y + best_mvy),0,codingOption.FrameHeight);
                            int fy = position((y - best_mvy),0,codingOption.FrameHeight);

                            for(int jj=0;jj<8;jj++){
                                int x = left + jj;
                                int idx = y*codingOption.FrameWidth + x;

                                int px = position((x + best_mvx),0,codingOption.FrameWidth);
                                int fx = position((x - best_mvx),0,codingOption.FrameWidth);
                                int past_idx = py*codingOption.FrameWidth + px;
                                int future_idx = fy*codingOption.FrameWidth + fx;

                                sideInfoFrame[idx] = (int)((double)reconsFrames[begin][past_idx] * interPC + (double)reconsFrames[end][future_idx] * interFC);

                            }                        
                        }
                    }
                }
            }

        }
    }
    free(dist2);

}



void FMCFI(int method, unsigned char *sideInfoFrame, int FrameIdx, int begin, int end, EncodedFrame *encodedFrame){
    int *dist=(int*)malloc(sizeof(int)*Block88Num);

    if(method == 0){
        // Average Interpolation and just copy
        //printf("Average Interpolation...\n");
        AverInter(sideInfoFrame, FrameIdx, begin, end, dist);
        /*int sum=0;
        for(int i=0;i<Block88Num;i++){
            sum += dist[i];
        }
        printf("Total SAD from AI = %d\n",sum);*/


    }
	else if(method == 1){
        // AI and Fast Search
        //printf("Average Interpolation & Fast Search...\n");
        AverInter(sideInfoFrame, FrameIdx, begin, end, dist);
        /*int sum=0;
        for(int i=0;i<Block88Num;i++){
            sum += dist[i];
        }
        printf("Total SAD from AI = %d\n",sum);*/

        FastSearch(sideInfoFrame, FrameIdx, begin, end, dist, encodedFrame, 0.5);
        /*sum = 0;
        for(int i=0;i<Block88Num;i++){
            sum += dist[i];
        }
        printf("Total SAD from FS = %d\n",sum);*/

    }
	else{
        errorMsg("FMCFI");
    }
    //memcpy(SIFrames[FrameIdx],sideInfoFrame,sizeof(imgpel)*FrameSize);
    free(dist);
}

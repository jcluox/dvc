#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "global.h"
#include "error.h"
#include "omp.h"
#ifndef max
	#define max(A,B) ( (A) > (B) ? (A):(B))
#endif
#ifndef min
	#define min(A,B) ( (A) < (B) ? (A):(B)) 
#endif
//修改自boris 
/*
   X   X   X   X   X   X

   X   X   X   X   X   X

   X   X   X a X   X   X
           b c
   X   X   X   X   X   X

   X   X   X   X   X   X

   X   X   X   X   X   X
   
*/
double FilterCoef[3][SQR_FILTER]={
    { 0.0       ,    0.0       ,    0.0       ,    0.0       ,    0.0       ,   0.0       ,
        0.0       ,    0.0       ,    0.0       ,    0.0       ,    0.0       ,   0.0       ,
        1.0/32.0  ,   -5.0/32.0  ,   20.0/32.0  ,   20.0/32.0  ,   -5.0/32.0  ,   1.0/32.0  ,
        0.0       ,    0.0       ,    0.0       ,    0.0       ,    0.0       ,   0.0       ,
        0.0       ,    0.0       ,    0.0       ,    0.0       ,    0.0       ,   0.0       ,
        0.0       ,    0.0       ,    0.0       ,    0.0       ,    0.0       ,   0.0
    }, // a_pos
    { 0.0       ,    0.0       ,    1.0/32.0  ,    0.0       ,    0.0       ,   0.0       ,
        0.0       ,    0.0       ,   -5.0/32.0  ,    0.0       ,    0.0       ,   0.0       ,
        0.0       ,    0.0       ,   20.0/32.0  ,    0.0       ,    0.0       ,   0.0       ,
        0.0       ,    0.0       ,   20.0/32.0  ,    0.0       ,    0.0       ,   0.0       ,
        0.0       ,    0.0       ,   -5.0/32.0  ,    0.0       ,    0.0       ,   0.0       ,
        0.0       ,    0.0       ,    1.0/32.0  ,    0.0       ,    0.0       ,   0.0
    }, // b_pos
    { 1.0/1024.0,   -5.0/1024.0,   20.0/1024.0,   20.0/1024.0,   -5.0/1024.0,   1.0/1024.0,
        -5.0/1024.0,   25.0/1024.0, -100.0/1024.0, -100.0/1024.0,   25.0/1024.0,  -5.0/1024.0,
        20.0/1024.0, -100.0/1024.0,  400.0/1024.0,  400.0/1024.0, -100.0/1024.0,  20.0/1024.0,
        20.0/1024.0, -100.0/1024.0,  400.0/1024.0,  400.0/1024.0, -100.0/1024.0,  20.0/1024.0,
        -5.0/1024.0,   25.0/1024.0, -100.0/1024.0, -100.0/1024.0,   25.0/1024.0,  -5.0/1024.0,
        1.0/1024.0,   -5.0/1024.0,   20.0/1024.0,   20.0/1024.0,   -5.0/1024.0,   1.0/1024.0
    } // c_pos
};

//FIR filter 修改自 boris
void FIR_filter(unsigned char *inFrame, int width, int height, unsigned char *outFrame){
	int width2 = width << 1;
	int height2 = height << 1;
	int temp_w = width2 + (IMG_PAD_SIZE << 2);
	int temp_h = height2 + (IMG_PAD_SIZE << 2);
	//int pos_x, pos_y;
	unsigned char *img_temp;
	int img_temp_size = temp_w * temp_h;
	img_temp = (unsigned char *)malloc(sizeof(unsigned char) * img_temp_size);
	memset((void *)img_temp, 0, sizeof(unsigned char) * img_temp_size);
	int height2_imgPadSize2 = height2 + IMG_PAD_SIZE*2;
	int width2_imgPadSize2 = width2 + IMG_PAD_SIZE*2;
	int img_temp_row = 0;
	for(int i=-IMG_PAD_SIZE*2; i<height2_imgPadSize2; i++){
		//int img_temp_row = (i + 2*IMG_PAD_SIZE) * temp_w;
		int y_sub2 = ( (i+2*IMG_PAD_SIZE)&1 ) << 1;
		for(int j=-IMG_PAD_SIZE*2; j<width2_imgPadSize2; j++){
			int x_sub = (j+2*IMG_PAD_SIZE)&1; //x_sub = (j+2*IMG_PAD_SIZE)%2;
			//int y_sub = (i+2*IMG_PAD_SIZE)&1; //y_sub = (i+2*IMG_PAD_SIZE)%2;
			int sub_pos = x_sub + y_sub2; // 0~3
			double sum = 0.0;
			//printf("%d\n",(i+2*IMG_PAD_SIZE)*temp_w+(j+2*IMG_PAD_SIZE));
			if(sub_pos){
                int FilterCoefRow = 0;
				for(int ii=0; ii<FILTER_SIZE; ii++){
					int pos_y = max(0, min(height-1, i/2-FILTER_OFFSET+ii ) );  // y integer position
					if(i<0 && pos_y > 0){
						//printf("i = %d, ii = %d, pos_y = %d\n",i,ii,pos_y);
						pos_y--;
					}
					for(int jj=0; jj<FILTER_SIZE; jj++){						
						int pos_x = max(0, min(width -1, j/2-FILTER_OFFSET+jj ) );  // x integer position
						if(j<0 && pos_x > 0){
							//printf("j = %d, j/2 = %d, jj = %d, pos_x = %d\n",j,j/2,jj,pos_x);
							pos_x--;
						}
						//printf("(%d %d) ",pos_y,pos_x);
						sum += ( FilterCoef[sub_pos-1][FilterCoefRow+jj] * inFrame[pos_y*width+pos_x] );
					}
					FilterCoefRow += FILTER_SIZE;
				}
				img_temp[img_temp_row + (j+2*IMG_PAD_SIZE)] = max(0, min(255, (int)(sum+0.5)));
				//img_temp[(i+2*IMG_PAD_SIZE)*temp_w+(j+2*IMG_PAD_SIZE)] = 0;
			}
			else{
				img_temp[img_temp_row + (j+2*IMG_PAD_SIZE)] =
					inFrame[max(0, min(height-1, i/2)) * width + (max(0, min(width-1, j/2)))];
			}
		}
		img_temp_row += temp_w;
	}	    

	int frameRow = 0;
	img_temp_row = 2*IMG_PAD_SIZE * temp_w;
	for(int i=0;i<height2;i++){
		for(int j=0;j<width2;j++){
			outFrame[frameRow + j] = img_temp[img_temp_row + (j+2*IMG_PAD_SIZE)];
		}
		frameRow += width2;
		img_temp_row += temp_w;
	}
	free(img_temp);	
}

void FIR_filter_OpenMP(unsigned char *inFrame, int width, int height, unsigned char *outFrame){
	int width2 = width << 1;
	int height2 = height << 1;
	int temp_w = width2 + (IMG_PAD_SIZE << 2);
	int temp_h = height2 + (IMG_PAD_SIZE << 2);
	//int pos_x, pos_y;
	unsigned char *img_temp;
	int img_temp_size = temp_w * temp_h;
	img_temp = (unsigned char *)malloc(sizeof(unsigned char) * img_temp_size);
	memset((void *)img_temp, 0, sizeof(unsigned char) * img_temp_size);
	int height2_imgPadSize2 = height2 + IMG_PAD_SIZE*2;
	int width2_imgPadSize2 = width2 + IMG_PAD_SIZE*2;
#pragma omp parallel for
	for(int i=-IMG_PAD_SIZE*2; i<height2_imgPadSize2; i++){
		int img_temp_row = (i + 2*IMG_PAD_SIZE) * temp_w;
		int y_sub2 = ( (i+2*IMG_PAD_SIZE)&1 ) << 1;
		for(int j=-IMG_PAD_SIZE*2; j<width2_imgPadSize2; j++){
			int x_sub = (j+2*IMG_PAD_SIZE)&1; //x_sub = (j+2*IMG_PAD_SIZE)%2;
			//int y_sub = (i+2*IMG_PAD_SIZE)&1; //y_sub = (i+2*IMG_PAD_SIZE)%2;
			int sub_pos = x_sub + y_sub2; // 0~3
			double sum = 0.0;
			//printf("%d\n",(i+2*IMG_PAD_SIZE)*temp_w+(j+2*IMG_PAD_SIZE));
			if(sub_pos){
                int FilterCoefRow = 0;
				for(int ii=0; ii<FILTER_SIZE; ii++){
					int pos_y = max(0, min(height-1, i/2-FILTER_OFFSET+ii ) );  // y integer position
					if(i<0 && pos_y > 0){
						//printf("i = %d, ii = %d, pos_y = %d\n",i,ii,pos_y);
						pos_y--;
					}
					for(int jj=0; jj<FILTER_SIZE; jj++){						
						int pos_x = max(0, min(width -1, j/2-FILTER_OFFSET+jj ) );  // x integer position
						if(j<0 && pos_x > 0){
							//printf("j = %d, j/2 = %d, jj = %d, pos_x = %d\n",j,j/2,jj,pos_x);
							pos_x--;
						}
						//printf("(%d %d) ",pos_y,pos_x);
						sum += ( FilterCoef[sub_pos-1][FilterCoefRow+jj] * inFrame[pos_y*width+pos_x] );
					}
					FilterCoefRow += FILTER_SIZE;
				}
				img_temp[img_temp_row + (j+2*IMG_PAD_SIZE)] = max(0, min(255, (int)(sum+0.5)));
				//img_temp[(i+2*IMG_PAD_SIZE)*temp_w+(j+2*IMG_PAD_SIZE)] = 0;
			}
			else{
				img_temp[img_temp_row + (j+2*IMG_PAD_SIZE)] =
					inFrame[max(0, min(height-1, i/2)) * width + (max(0, min(width-1, j/2)))];
			}
		}
	}	    

	int frameRow = 0;
	int img_temp_row = 2*IMG_PAD_SIZE * temp_w;
	for(int i=0;i<height2;i++){
		for(int j=0;j<width2;j++){
			outFrame[frameRow + j] = img_temp[img_temp_row + (j+2*IMG_PAD_SIZE)];
		}
		frameRow += width2;
		img_temp_row += temp_w;
	}
	free(img_temp);	
}

void bilinear_filter(unsigned char *inFrame, int width, int height, unsigned char *outFrame){
	int width2 = width << 1;
	int height2 = height << 1;	
	//bilinear filter
	for(int i=0; i<height2; i++){
		double y = (double)i / 2.0;
		double a = y - floor(y);
		int ori_y = (int)y;
		int ori_y_add1 = ori_y + 1;
		if(ori_y_add1 >= height)
			ori_y_add1--;
		int ori_yRow = ori_y * width;
		int ori_y_add1Row = ori_y_add1 * width;
		int yRow = i * width2;
		for(int j=0; j<width2; j++){
			double x = (double)j / 2.0;
			double b = x - floor(x);
			int ori_x = (int)x;
			int ori_x_add1 = ori_x + 1; 
			if(ori_x_add1 >= width)
				ori_x_add1--;		
			
			outFrame[yRow + j] = (unsigned char)( 
				b*(1-a)*((double)inFrame[ori_yRow + ori_x_add1])
				+ b*a*((double)inFrame[ori_y_add1Row + ori_x_add1]) 
				+ (1-b)*(1-a)*((double)inFrame[ori_yRow + ori_x]) 
				+ (1-b)*a*((double)inFrame[ori_y_add1Row + ori_x])
				);
		}
	}
}

void lowPassFilter(unsigned char *inFrame, int width, int height, unsigned char *outFrame){	
	int frameRow = 0;
	for(int i=0; i<height; i++){
		//int frameRow = i*width;
		for(int j=0; j<width; j++){
			double sum = 0; 
			//3x3 mean filter
			for(int m=i-1; m<=i+1; m++){
				if(m<0 || m>=height)
					continue;
				for(int n=j-1;n<=j+1;n++){
					if(n<0 || n>=width)
						continue;
					sum += (double)inFrame[m*width+n];
				}
			}
			outFrame[frameRow+j] = (unsigned char)(sum/9);
		}
		frameRow += width;
	}
}
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "transform.h"

double postScale[BLOCKSIZE44];
double preScale[BLOCKSIZE44];

 //used by transform of 264 Intra
int MF[BLOCKSIZE44];
int *qp_rem_matrix;
int *qp_per_matrix;

void matrix_mult_d(double *A,double *B,double *C){
    for(int i=0;i<4;i++){
        for(int j=0;j<4;j++){
            for(int k=0;k<4;k++){
                C[(i<<2)+j] += A[(i<<2)+k] * B[(k<<2)+j];
            }
        }
    } 
}

void initTransform(){
	double a = 0.5;
	double b = sqrt(0.4);
	double a2 = 0.25, b2 = 0.4, ab = a*b;

	postScale[0] = a2;
	postScale[1] = ab/2;
	postScale[2] = a2;
	postScale[3] = ab/2;
	postScale[4] = ab/2;
	postScale[5] = b2/4;
	postScale[6] = ab/2;
	postScale[7] = b2/4;
	postScale[8] = a2;
	postScale[9] = ab/2;
	postScale[10] = a2;
	postScale[11] = ab/2;
	postScale[12] = ab/2;
	postScale[13] = b2/4;
	postScale[14] = ab/2;
	postScale[15] = b2/4;

	preScale[0] = a2;
	preScale[1] = ab;
	preScale[2] = a2;
	preScale[3] = ab;
	preScale[4] = ab;
	preScale[5] = b2;
	preScale[6] = ab;
	preScale[7] = b2;
	preScale[8] = a2;
	preScale[9] = ab;
	preScale[10] = a2;
	preScale[11] = ab;
	preScale[12] = ab;
	preScale[13] = b2;
	preScale[14] = ab;
	preScale[15] = b2;

	for(int i=0; i<BLOCKSIZE44; i++)
		preScale[i] *= 64.0;

	//Boris
	if(codingOption.EnableIntraMode){
        // calculate K
        Kmatrix = (double *)malloc(sizeof(double)*BLOCKSIZE44);
        // memset(Kmatrix,0,sizeof(double)*BLOCKSIZE44);
        double rho = 0.6;
        double Rx[BLOCKSIZE44] = 
        {   1.0,        rho,    rho*rho,  rho*rho*rho,
            rho,        1.0,        rho,      rho*rho,
            rho*rho,    rho,        1.0,          rho,
            rho*rho*rho,rho*rho,   rho,          1.0};
        //printf("Rx:\n");
        //printBlock44_d(Rx);
        double C[BLOCKSIZE44] = {
			1.0, 1.0, 1.0, 1.0,
            2.0, 1.0,-1.0,-2.0,
            1.0,-1.0,-1.0, 1.0,
            1.0,-2.0, 2.0,-1.0 
		};
        //printf("C:\n");
        //printBlock44_d(C);
        double CT[BLOCKSIZE44];
        for(int i=0;i<4;i++){
            for(int j=0;j<4;j++){
                CT[j*4+i] = C[i*4+j]; 
            }
        }

        double tempK[BLOCKSIZE44];
        double tempK2[BLOCKSIZE44];

        memset(tempK, 0, sizeof(double) * BLOCKSIZE44);
        memset(tempK2, 0, sizeof(double) * BLOCKSIZE44);

        matrix_mult_d(C, Rx, tempK);

        matrix_mult_d(tempK, CT, tempK2);

        for(int i=0; i<4; i++){
            for(int j=0; j<4; j++){
                Kmatrix[i*4+j] = tempK2[i*4+i] * tempK2[j*4+j] * postScale[i*4+j] * postScale[i*4+j];
            }
        } 
        //printf("K:\n");
        //printBlock44_d(Kmatrix);
        
        qp_rem_matrix = (int*)malloc(sizeof(int)*(MAX_QP+1));
        qp_per_matrix = (int*)malloc(sizeof(int)*(MAX_QP+1));
        for(int i=0;i<(MAX_QP+1);i++){
            qp_rem_matrix[i] = i % 6;
            qp_per_matrix[i] = i / 6;
        }

        // MF
        int qbits = (int)(15.0 + floor((double)codingOption.IntraQP / 6.0));
        double Qstep = pow( 2.0,((double)(codingOption.IntraQP - 4) / 6.0));
        //printf("qbits = %d, Qstep = %lf\n",qbits,Qstep);
        for(int i=0;i<4;i++){
            for(int j=0;j<4;j++){
                MF[(i<<2)+j] = ROUND(((double)(1<<qbits) * postScale[(i<<2)+j]) / Qstep);
            }    
        }
        /*printf("MF:\n");
        for(int i=0;i<4;i++){
            for(int j=0;j<4;j++){
                printf("%5d ",MF[(i<<2)+j]);
            }
            printf("\n");
        }*/
    }

}

void forward44(int *blk){
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

	//C*X*C'.*postScale
	for(int i=0; i<BLOCKSIZE44; i++)
		blk[i] = ROUND( ((double)blk[i])*postScale[i] );
}

void inverse44(int *blk){
	int tmpblk[BLOCKSIZE44];
	//preScale
	for(int i=0; i<BLOCKSIZE44; i++)
		blk[i] = ROUND( ((double)blk[i])*preScale[i] );

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
void forwardTransform(unsigned char *oriFrame, int *transFrame, EncodedFrame *encodedFrame){
	//initial AC coefficient range
	for(int i=0; i<BLOCKSIZE44; i++)
		encodedFrame->ACRange[i] = 0;

	int blkIdx = 0;
	for(int i=0; i<Block88Height; i++){
		int top = i << 3;
		for(int j=0; j<Block88Width; j++){
			//for each block
			if(encodedFrame->skipBlockFlag[blkIdx++] == 0){
				int left = j << 3;

				//forward transform
				int block[BLOCKSIZE44];
				for(int ii=0; ii<8; ii+=4){
					for(int jj=0; jj<8; jj+=4){
						
						for(int m=0; m<4; m++){
							int idx1 = m << 2;
							int idx2 = (top + ii + m) * (codingOption.FrameWidth) + left + jj;
							for(int n=0; n<4; n++)
								block[idx1 + n] = (int)oriFrame[idx2 + n];			
						}				
						
						//transform
						forward44(block);

						for(int m=0; m<4; m++){
							int idx1 = m << 2;
							int idx2 = (top + ii + m) * (codingOption.FrameWidth) + left + jj;
							for(int n=0; n<4; n++){
								int idx = idx1 + n;
								transFrame[idx2 + n] = block[idx];	
								if(m!=0 || n!=0){
									//calculate dynamic AC coefficient range (use for quantization)
									if(abs(block[idx]) > encodedFrame->ACRange[idx])
										encodedFrame->ACRange[idx] = abs(block[idx]);
								}
							}		
						}

					}
				}
			}
		}
	}
}

void forwardTransform_Pure(unsigned char *oriFrame, int *transFrame, EncodedFrame *encodedFrame){

	int blkIdx = 0;
	for(int i=0; i<Block88Height; i++){
		int top = i << 3;
		for(int j=0; j<Block88Width; j++){
			//for each block
			if(encodedFrame->skipBlockFlag[blkIdx] == 0 && encodedFrame->intraBlockFlag[blkIdx] == 0){
				int left = j << 3;

				//forward transform
				int block[BLOCKSIZE44];
				for(int ii=0; ii<8; ii+=4){
					for(int jj=0; jj<8; jj+=4){
						
						for(int m=0; m<4; m++){
							int idx1 = m << 2;
							int idx2 = (top + ii + m) * (codingOption.FrameWidth) + left + jj;
							for(int n=0; n<4; n++)
								block[idx1 + n] = (int)oriFrame[idx2 + n];			
						}				
						
						//transform
						forward44(block);

						for(int m=0; m<4; m++){
							int idx1 = m << 2;
							int idx2 = (top + ii + m) * (codingOption.FrameWidth) + left + jj;
							for(int n=0; n<4; n++){
								int idx = idx1 + n;
								transFrame[idx2 + n] = block[idx];
							}		
						}

					}
				}
			}
			blkIdx++;
		}
	}
}


void inverseTransform(int *transFrame, unsigned char *refFrame, unsigned char *reconsFrame, EncodedFrame *encodedFrame){
	int blkIdx = 0;
	for(int i=0; i<Block88Height; i++){
		int top = i << 3;
		for(int j=0; j<Block88Width; j++){
			//for each block
			int left = j << 3;

			if(encodedFrame->skipBlockFlag[blkIdx] == 1){
				//skip block
				//copy co-located block from reference frame
				for(int m=0; m<8; m++){
					int idx = (top + m) * (codingOption.FrameWidth) + left;
					for(int n=0; n<8; n++){						
						reconsFrame[idx + n] = refFrame[idx + n];
					}
				}
			}
			else if(encodedFrame->intraBlockFlag[blkIdx] == 0){
				//WZ block
				//inverse transform
				int block[BLOCKSIZE44];
				for(int ii=0; ii<8; ii+=4){
					for(int jj=0; jj<8; jj+=4){
						int idx1 = 0;
						for(int m=0; m<4; m++){
							//int idx1 = m << 2;
							int idx2 = (top + ii + m) * (codingOption.FrameWidth) + left + jj;
							for(int n=0; n<4; n++)
								block[idx1++] = transFrame[idx2 + n];			
						}				
						
						//transform
						inverse44(block);
						
						idx1 = 0;
						for(int m=0; m<4; m++){
							//int idx1 = m << 2;
							int idx2 = (top + ii + m) * (codingOption.FrameWidth) + left + jj;
							for(int n=0; n<4; n++){
								int coeff = block[idx1++];
								if(coeff < 0)
									reconsFrame[idx2 + n] = (unsigned char)0;
								else if(coeff > 255)
									reconsFrame[idx2 + n] = (unsigned char)255;
								else
									reconsFrame[idx2 + n] = (unsigned char)coeff;
							}		
						}

					}
				}
			}
			blkIdx++;
		}
	}
}

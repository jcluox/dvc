#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "LdpcaDecoder.h"
#ifdef CUDA
#include "LDPCA_cuda.h"
#endif
#include "global.h"
#include "error.h"
#include "ReadLdpcaLadderFile_LdpcaDecoder.h"
#include "omp.h"
#include "crc8.h"

//#define LOG2 log((float)2)
#define LOG2 (0.693147181f)
#define JUMPLEN 4
              // request: 1  2  3  4 .... 
int jump_step[JUMPLEN] = {0, 1, 2, 3}; //24,48,72,96

unsigned char *Syndrome;
char *Buf, *Check_LLR;
float *LLR_extrinsic, *Check_LLR_mag, *LLR_overall;

#define NEGATIVE_EARLY_JUMP
#define POSITIVE_EARLY_JUMP

extern int LDPC_iterations;
//#define MAX_ITERATIONS 100

//修改自小小白
////////////////////////////////////////////////////////////////////
// <BitToByte>
//	input:
//		Bits, BitLength 
//	output:
//		unsigned char*, numBytes
////////////////////////////////////////////////////////////////////
unsigned char* BitToByte(unsigned char *Bits, int BitLength, int *numBytes)
{	
	int i, remainder;
	int ByteLength;
	unsigned char *Bytes;
	
	ByteLength = BitLength >> 3; //BitLength / 8
	remainder = BitLength & 7;	//BitLength % 8
	if(remainder != 0)
		ByteLength++;

	Bytes = (unsigned char *)malloc(sizeof(unsigned char) * ByteLength);	
	*numBytes = ByteLength;	//output number of bytes

	int byteIdx = 0;
	unsigned char tmp = 0;
	for(i=0; i<BitLength; i++){
		tmp |= Bits[i];
		if(((i+1)&7) == 0){	//(i+1)%8 == 0
			Bytes[byteIdx++] = tmp;
			tmp = 0;
		}
		else
			tmp <<= 1;
	}
	if(remainder != 0)
		Bytes[byteIdx++] = tmp<<(7-remainder);

	return Bytes;
}

////////////////////////////////////////////////////////////////////
// <ByteToBit>
//	input:
//		Bytes: store 8 bit infomation in each byte
//		ByteLength: length of Bytes
//		SourceLength: used for checking only
//	output:
//		unsigned char*
////////////////////////////////////////////////////////////////////
unsigned char* ByteToBit(unsigned char *Bytes, int ByteLength, int SourceLength)
{
	//TimeStamp ts,te; timeStart(&ts);
	unsigned char *Bits;
	int i;
	
	if((ByteLength<<3) < SourceLength)
		errorMsg("In function ByteToBit: SourceLength is larger than ByteLength");

	Bits = (unsigned char *)malloc(sizeof(unsigned char) * SourceLength);

	int byteIdx = 0;
	unsigned char tmp;
	//#pragma omp parallel for
	for(i=0; i<SourceLength; i++){
	//	int idx = i>>3;
	//	int shift = 7 - (i&0x07);
      //  Bits[i] = (Bytes[idx]>>shift) & 1;
		if((i&7) == 0)	//i%8 == 0
			tmp = Bytes[byteIdx++];
		else
			tmp <<= 1;
		Bits[i] = tmp >> 7;
	}
	//byte2bitTime += timeElapse(ts, &te);
	return Bits;
}

float ComputeCE_OpenMP(float *prob)	//prob = p0/p1
{
	float RateLowerBound = 0.0f;
	//compute p0 or p1, (Pco = CrossOver Prob)
	#pragma omp parallel for reduction(+:RateLowerBound)
	for(int k=0;k<codeLength;k++){
		float Pco = prob[k]/(1.0f + prob[k]);

		//H(X|Y)
		float tmp = log(1-Pco);
		RateLowerBound += ( Pco*(-log(Pco)+tmp) - tmp ) / LOG2;
	}
	RateLowerBound /= (float)codeLength;
		
	//printf("Conditional Entropy = %lf\n", RateLowerBound);
	
	return RateLowerBound;
}
float ComputeCE(float *prob)	//prob = p0/p1
{
	float RateLowerBound = 0.0f;
	//compute p0 or p1, (Pco = CrossOver Prob)	
	for(int k=0;k<codeLength;k++){
		float Pco = prob[k]/(1.0f + prob[k]);

		//H(X|Y)
		float tmp = log(1-Pco);
		RateLowerBound += ( Pco*(-log(Pco)+tmp) - tmp ) / LOG2;
	}
	RateLowerBound /= (float)codeLength;
		
	//printf("Conditional Entropy = %lf\n", RateLowerBound);
	
	return RateLowerBound;
}

void initLdpcaBuffer(){
	Syndrome = (unsigned char *)malloc(sizeof(unsigned char) * codeLength); 

	LLR_extrinsic = (float *)malloc(sizeof(float) * numEdges);
	Check_LLR = (char *)malloc(sizeof(char) * numEdges);
	Check_LLR_mag = (float *)malloc(sizeof(float) * numEdges);
	Buf = (char *)malloc(sizeof(char) * numEdges);
	LLR_overall = (float *)malloc(sizeof(float) * codeLength);
}

void freeLdpcaBuffer(){
	free(Syndrome);	

	free(LLR_extrinsic);
	free(Check_LLR);
	free(Buf);
	free(Check_LLR_mag);
	free(LLR_overall);
}
int beliefPropagation_OpenMP(float *LLR_intrinsic, unsigned char *syndrome, unsigned char *decoded, int syndromeLength, int nCode)
{
    int iteration;
    //double *LLR_extrinsic, *check_LLR_mag, *rowTotal, *LLR_overall;
	int CheckNodeWrong;
	int *pIrReadPtr = ir[nCode];
    
	/*LLR_extrinsic = (double *)malloc(sizeof(double) * numEdges);
	//check_LLR = (double *)malloc(sizeof(double) * numEdges);
	char *check_LLR = (char *)malloc(sizeof(char) * numEdges);
	check_LLR_mag = (double *)malloc(sizeof(double) * numEdges);
	char *buf = (char *)malloc(sizeof(char) * numEdges);
	LLR_overall = (double *)malloc(sizeof(double) * codeLength);*/
	float *rowTotal = (float *)malloc(sizeof(float) * syndromeLength);

    
    //initialize variable-to-check messages
    for(int k=0; k<codeLength; k++)
        for(int l=jc[k]; l<jc[k+1]; l++)
            LLR_extrinsic[l] = LLR_intrinsic[k];
    
	int sameCount = 0;
    for(iteration=0; iteration < MAX_ITERATIONS; iteration++, LDPC_iterations++)
    {
        //Step 1: compute check-to-variable messages
        //printf("iter = %d\n", iteration);
        for(int k=0; k<numEdges; k++) //1188
        {
            //check_LLR[k] = (double) ((LLR_extrinsic[k]<0) ? -1 : 1);
			Check_LLR[k] = (char)(( (*( (int*)&(LLR_extrinsic[k]) )) >> 31 ) | 1); //1 or -1
            //check_LLR_mag[k] = ((LLR_extrinsic[k]<0) ? -LLR_extrinsic[k] : LLR_extrinsic[k]);
			Check_LLR_mag[k] = (float)Check_LLR[k] * LLR_extrinsic[k]; //abs(LLR_extrinsic[k])
        }
        
        for(int k=0; k<syndromeLength; k++)
			Buf[k] = 1 - (char)(syndrome[k] << 1);
            //rowTotal[k] = (double) ((syndrome[k]==1) ? -1 : 1);  // (1-2s)

		for(int k=0; k<numEdges; k++){
			Buf[pIrReadPtr[k]] = (Buf[pIrReadPtr[k]] ^ Check_LLR[k]) | 1;  //1 or -1
			//rowTotal[pIrReadPtr[k]] *= check_LLR[k];   // 與對應到的check node相乘
		}
        
        for(int k=0; k<numEdges; k++)
            Check_LLR[k] = (Check_LLR[k] ^ Buf[pIrReadPtr[k]]) | 1;
			//check_LLR[k] = check_LLR[k] * (char)rowTotal[pIrReadPtr[k]]; // 與對應到的check node相乘
            //sign of check-to-variable messages
        
		for(int k=0; k<syndromeLength; k++)
            rowTotal[k] = 0;

#pragma omp parallel for
		for(int k=0; k<numEdges; k++)
            Check_LLR_mag[k] = -log( tanh( (Check_LLR_mag[k] + (float)0.000000001)/2 ) );
        for(int k=0; k<numEdges; k++)
            rowTotal[pIrReadPtr[k]] += Check_LLR_mag[k];

#pragma omp parallel for
        for(int k=0; k<numEdges; k++)
            Check_LLR_mag[k] = -log( tanh( (rowTotal[pIrReadPtr[k]] - Check_LLR_mag[k] + (float)0.000000001)/2 ) ) * (float)Check_LLR[k];
            //magnitude of check-to-variable messages
            
        //for(k=0; k<numEdges; k++)
          //  check_LLR_mag[k] *= (double)check_LLR[k];
            //check-to-variable messages : check_LLR[k]
            
        //Step 2: compute variable-to-check messages
        for(int k=0; k<codeLength; k++)
        {
			float tmp = LLR_intrinsic[k];
            //LLR_overall[k] = LLR_intrinsic[k];
            for(int l=jc[k]; l<jc[k+1]; l++)
				tmp += Check_LLR_mag[l];
                //LLR_overall[k] += check_LLR_mag[l];

			for(int l=jc[k]; l<jc[k+1]; l++)
                LLR_extrinsic[l] = tmp - Check_LLR_mag[l];

			LLR_overall[k] = tmp;
        }
            
        //for(k=0; k<codeLength; k++)
          //  for(l=jc[k]; l<jc[k+1]; l++)
            //    LLR_extrinsic[l] = LLR_overall[k] - check_LLR_mag[l];
                //variable-to-check messages
            
        //Step 3: test convergence and syndrome condition
		#ifndef NEGATIVE_EARLY_JUMP
			for(int k=0; k<codeLength; k++)
				decoded[k] = (unsigned char)(( (*( (int*)&(LLR_overall[k]) )) >> 31 ) & 1); //((LLR_overall[k]<0.0) ? 1 : 0);
		#else
			int l = 0;
			for(int k=0; k<codeLength; k++)
				if(decoded[k] == ((LLR_overall[k]<0) ? 1 : 0))
					l++;
				else
					decoded[k] = ((LLR_overall[k]<0) ? 1 : 0);        
			sameCount = ((l==codeLength) ? sameCount+1 : 0); 
			if(sameCount==4)
				return 0; //convergence (to wrong answer)
		#endif
		
        for(int k=0; k<syndromeLength; k++)
            rowTotal[k] = syndrome[k];
        for(int k=0; k<codeLength; k++)
            for(int l=jc[k]; l<jc[k+1]; l++)
                rowTotal[pIrReadPtr[l]] += (float)decoded[k];
		#ifdef POSITIVE_EARLY_JUMP
			CheckNodeWrong = 0;
			for(int k=0; k<syndromeLength; k++)
				if( (((int)rowTotal[k]) & 1) != 0)
				{
					CheckNodeWrong++;	//之後做 Early Stop 會用到
										//利用check node equation滿足幾條來確定要不要提早結束
					break;
				}       

			if(CheckNodeWrong == 0){
				return 1; //all syndrome checks satisfied
			}
		#endif
    } //end iteration 100
	//#ifndef POSITIVE_EARLY_JUMP
	//	#ifndef NEGATIVE_EARLY_JUMP
			for(int k=0; k<syndromeLength; k++)
				if( (((int)rowTotal[k]) & 1) != 0){
					free(rowTotal);
					return 0;
				}
			free(rowTotal);
			return 1; //all syndrome checks satisfied
	//	#endif
	//#endif

	/*free(LLR_extrinsic);
	free(check_LLR);
	free(buf);
	free(check_LLR_mag);
	free(LLR_overall);*/
	free(rowTotal);

    return 0;
}

//written by 多多.
int minsum_beliefPropagation(float *LLR_intrinsic, unsigned char *syndrome, unsigned char *decoded, int syndromeLength, int nCode)
{
    int iteration;
    //double *LLR_extrinsic, *check_LLR_mag, *rowTotal, *LLR_overall;
	//int CheckNodeWrong;
	int *pIrReadPtr = ir[nCode];
    
	/*LLR_extrinsic = (double *)malloc(sizeof(double) * numEdges);
	//check_LLR = (double *)malloc(sizeof(double) * numEdges);
	char *check_LLR = (char *)malloc(sizeof(char) * numEdges);
	check_LLR_mag = (double *)malloc(sizeof(double) * numEdges);
	char *buf = (char *)malloc(sizeof(char) * numEdges);
	LLR_overall = (double *)malloc(sizeof(double) * codeLength);*/
	float *rowTotal = (float *)malloc(sizeof(float) * syndromeLength);
	float *rowTotal2 = (float *)malloc(sizeof(float) * syndromeLength);

    
    //initialize variable-to-check messages
    for(int k=0; k<codeLength; k++)
        for(int l=jc[k]; l<jc[k+1]; l++)
            LLR_extrinsic[l] = LLR_intrinsic[k];
	int sameCount = 0;    
    for(iteration=0; iteration < MAX_ITERATIONS; iteration++)
    {
        //Step 1: compute check-to-variable messages
        //printf("iter = %d\n", iteration);
        for(int k=0; k<numEdges; k++) //1188
        {
            //check_LLR[k] = (double) ((LLR_extrinsic[k]<0) ? -1 : 1);
			Check_LLR[k] = (char)(( (*( (int*)&(LLR_extrinsic[k]) )) >> 31 ) | 1); //1 or -1
            //check_LLR_mag[k] = ((LLR_extrinsic[k]<0) ? -LLR_extrinsic[k] : LLR_extrinsic[k]);
			Check_LLR_mag[k] = (float)Check_LLR[k] * LLR_extrinsic[k]; //abs(LLR_extrinsic[k])
        }
        
        for(int k=0; k<syndromeLength; k++)
			Buf[k] = 1 - (char)(syndrome[k] << 1);
            //rowTotal[k] = (double) ((syndrome[k]==1) ? -1 : 1);  // (1-2s)

		for(int k=0; k<numEdges; k++){
			Buf[pIrReadPtr[k]] = (Buf[pIrReadPtr[k]] ^ Check_LLR[k]) | 1;  //1 or -1
			//rowTotal[pIrReadPtr[k]] *= check_LLR[k];   // 與對應到的check node相乘
		}
        
        for(int k=0; k<numEdges; k++)
            Check_LLR[k] = (Check_LLR[k] ^ Buf[pIrReadPtr[k]]) | 1;
			//check_LLR[k] = check_LLR[k] * (char)rowTotal[pIrReadPtr[k]]; // 與對應到的check node相乘
            //sign of check-to-variable messages
        
		for(int k=0; k<syndromeLength; k++){
            rowTotal[k] = (float)HUGE_VAL;
			rowTotal2[k] = (float)HUGE_VAL;
		}
		//find min and second min
		for(int k=0; k<numEdges; k++){
			if(rowTotal[pIrReadPtr[k]] > Check_LLR_mag[k]){
				rowTotal2[pIrReadPtr[k]] = rowTotal[pIrReadPtr[k]];  //second min
				rowTotal[pIrReadPtr[k]] = Check_LLR_mag[k];  //min
			}
			else if(rowTotal2[pIrReadPtr[k]] > Check_LLR_mag[k]){
				rowTotal2[pIrReadPtr[k]] = Check_LLR_mag[k];  //second min
			}
		}	

		for(int k=0; k<numEdges; k++){
			if(Check_LLR_mag[k] == rowTotal[pIrReadPtr[k]])  //自己就是最小的那個.
				Check_LLR_mag[k] = (float)Check_LLR[k] * rowTotal2[pIrReadPtr[k]]/1.25f; //取第二小的.
			else
				Check_LLR_mag[k] = (float)Check_LLR[k] * rowTotal[pIrReadPtr[k]]/1.25f; //取最小.
		}	
            //magnitude of check-to-variable messages
            
        //for(k=0; k<numEdges; k++)
          //  check_LLR_mag[k] *= (double)check_LLR[k];
            //check-to-variable messages : check_LLR[k]
            
        //Step 2: compute variable-to-check messages
        for(int k=0; k<codeLength; k++)
        {
			float tmp = LLR_intrinsic[k];
            //LLR_overall[k] = LLR_intrinsic[k];
            for(int l=jc[k]; l<jc[k+1]; l++)
				tmp += Check_LLR_mag[l];
                //LLR_overall[k] += check_LLR_mag[l];

			for(int l=jc[k]; l<jc[k+1]; l++)
                LLR_extrinsic[l] = tmp - Check_LLR_mag[l];

			LLR_overall[k] = tmp;
        }
            
        //for(k=0; k<codeLength; k++)
          //  for(l=jc[k]; l<jc[k+1]; l++)
            //    LLR_extrinsic[l] = LLR_overall[k] - check_LLR_mag[l];
                //variable-to-check messages
            
        //Step 3: test convergence and syndrome condition
  //===================SSW====================
        //for(int k=0; k<codeLength; k++)
          //  decoded[k] = (unsigned char)(( (*( (int*)&(LLR_overall[k]) )) >> 31 ) & 1); //((LLR_overall[k]<0.0) ? 1 : 0);
  ///*===================negative early jump out====================
  //      int l = 0;
  //      for(int k=0; k<codeLength; k++)
  //          if(decoded[k] == ((LLR_overall[k]<0) ? 1 : 0))
  //              l++;
  //          else
  //              decoded[k] = ((LLR_overall[k]<0) ? 1 : 0);        
  //      sameCount = ((l==codeLength) ? sameCount+1 : 0); 
		//if(sameCount==5)
  //          return 0; //convergence (to wrong answer)
  //===========================================================*/
  
		
        for(int k=0; k<syndromeLength; k++)
            rowTotal[k] = syndrome[k];
        for(int k=0; k<codeLength; k++)
            for(int l=jc[k]; l<jc[k+1]; l++)
                rowTotal[pIrReadPtr[l]] += (float)decoded[k];
                
		//positive early jump out
		//CheckNodeWrong = 0;
  //      for(int k=0; k<syndromeLength; k++)
  //          if( (((int)rowTotal[k]) & 1) != 0)
		//	{
  //              CheckNodeWrong++;	//之後做 Early Stop 會用到
		//							//利用check node equation滿足幾條來確定要不要提早結束
		//		break;
		//	}       

		//if(CheckNodeWrong == 0){
		//	return 1; //all syndrome checks satisfied
		//}
           
    }
    //================no jump==============
    for(int k=0; k<syndromeLength; k++)
		if( (((int)rowTotal[k]) & 1) != 0){
			free(rowTotal);
			return 0;
		}
	free(rowTotal);
	return 1; //all syndrome checks satisfied
	//=====================================


	/*free(LLR_extrinsic);
	free(check_LLR);
	free(buf);
	free(check_LLR_mag);
	free(LLR_overall);*/
	free(rowTotal);

    return 0;
}
int beliefPropagation(float *LLR_intrinsic, unsigned char *syndrome, unsigned char *decoded, int syndromeLength, int nCode)
{
    int iteration;
    //double *LLR_extrinsic, *check_LLR_mag, *rowTotal, *LLR_overall;
	int CheckNodeWrong;
	int *pIrReadPtr = ir[nCode];
    
	/*LLR_extrinsic = (double *)malloc(sizeof(double) * numEdges);
	//check_LLR = (double *)malloc(sizeof(double) * numEdges);
	char *check_LLR = (char *)malloc(sizeof(char) * numEdges);
	check_LLR_mag = (double *)malloc(sizeof(double) * numEdges);
	char *buf = (char *)malloc(sizeof(char) * numEdges);
	LLR_overall = (double *)malloc(sizeof(double) * codeLength);*/
	float *rowTotal = (float *)malloc(sizeof(float) * syndromeLength);

    
    //initialize variable-to-check messages
    for(int k=0; k<codeLength; k++)
        for(int l=jc[k]; l<jc[k+1]; l++)
            LLR_extrinsic[l] = LLR_intrinsic[k];
    
    for(iteration=0; iteration < MAX_ITERATIONS; iteration++)
    {
        //Step 1: compute check-to-variable messages
        //printf("iter = %d\n", iteration);
        for(int k=0; k<numEdges; k++) //1188
        {
            //check_LLR[k] = (double) ((LLR_extrinsic[k]<0) ? -1 : 1);
			Check_LLR[k] = (char)(( (*( (int*)&(LLR_extrinsic[k]) )) >> 31 ) | 1); //1 or -1
            //check_LLR_mag[k] = ((LLR_extrinsic[k]<0) ? -LLR_extrinsic[k] : LLR_extrinsic[k]);
			Check_LLR_mag[k] = (float)Check_LLR[k] * LLR_extrinsic[k]; //abs(LLR_extrinsic[k])
        }
        
        for(int k=0; k<syndromeLength; k++)
			Buf[k] = 1 - (char)(syndrome[k] << 1);
            //rowTotal[k] = (double) ((syndrome[k]==1) ? -1 : 1);  // (1-2s)

		for(int k=0; k<numEdges; k++){
			Buf[pIrReadPtr[k]] = (Buf[pIrReadPtr[k]] ^ Check_LLR[k]) | 1;  //1 or -1
			//rowTotal[pIrReadPtr[k]] *= check_LLR[k];   // 與對應到的check node相乘
		}
        
        for(int k=0; k<numEdges; k++)
            Check_LLR[k] = (Check_LLR[k] ^ Buf[pIrReadPtr[k]]) | 1;
			//check_LLR[k] = check_LLR[k] * (char)rowTotal[pIrReadPtr[k]]; // 與對應到的check node相乘
            //sign of check-to-variable messages
        
		for(int k=0; k<syndromeLength; k++)
            rowTotal[k] = 0;
//#pragma omp parallel for
		for(int k=0; k<numEdges; k++)
            Check_LLR_mag[k] = -log( tanh( (Check_LLR_mag[k] + (float)0.000000001)/2 ) );
        for(int k=0; k<numEdges; k++)
            rowTotal[pIrReadPtr[k]] += Check_LLR_mag[k];

//#pragma omp parallel for
        for(int k=0; k<numEdges; k++)
            Check_LLR_mag[k] = -log( tanh( (rowTotal[pIrReadPtr[k]] - Check_LLR_mag[k] + (float)0.000000001)/2 ) ) * (float)Check_LLR[k];
            //magnitude of check-to-variable messages
            
        //for(k=0; k<numEdges; k++)
          //  check_LLR_mag[k] *= (double)check_LLR[k];
            //check-to-variable messages : check_LLR[k]
            
        //Step 2: compute variable-to-check messages
        for(int k=0; k<codeLength; k++)
        {
			float tmp = LLR_intrinsic[k];
            //LLR_overall[k] = LLR_intrinsic[k];
            for(int l=jc[k]; l<jc[k+1]; l++)
				tmp += Check_LLR_mag[l];
                //LLR_overall[k] += check_LLR_mag[l];

			for(int l=jc[k]; l<jc[k+1]; l++)
                LLR_extrinsic[l] = tmp - Check_LLR_mag[l];

			LLR_overall[k] = tmp;
        }
            
        //for(k=0; k<codeLength; k++)
          //  for(l=jc[k]; l<jc[k+1]; l++)
            //    LLR_extrinsic[l] = LLR_overall[k] - check_LLR_mag[l];
                //variable-to-check messages
            
        //Step 3: test convergence and syndrome condition
        for(int k=0; k<codeLength; k++)
            decoded[k] = (unsigned char)(( (*( (int*)&(LLR_overall[k]) )) >> 31 ) & 1); //((LLR_overall[k]<0.0) ? 1 : 0);
  
		
        for(int k=0; k<syndromeLength; k++)
            rowTotal[k] = syndrome[k];
        for(int k=0; k<codeLength; k++)
            for(int l=jc[k]; l<jc[k+1]; l++)
                rowTotal[pIrReadPtr[l]] += (float)decoded[k];
                
		CheckNodeWrong = 0;
        for(int k=0; k<syndromeLength; k++)
            if( (((int)rowTotal[k]) & 1) != 0)
			{
                CheckNodeWrong++;	//之後做 Early Stop 會用到
									//利用check node equation滿足幾條來確定要不要提早結束
				break;
			}       

		if(CheckNodeWrong == 0){
			return 1; //all syndrome checks satisfied
		}
           
    }
    
	/*free(LLR_extrinsic);
	free(check_LLR);
	free(buf);
	free(check_LLR_mag);
	free(LLR_overall);*/
	free(rowTotal);

    return 0;
}

#define BIT_LEN2INT_LEN(x) ( ((x)&31)==0? ( (x)>>5 ) : ( ( (x)>>5 ) + 1) )

int* BitToWord(unsigned char *Bits, int BitLength, int *numWords)
{	
	int i, remainder;
	int ByteLength;
	unsigned char *Bytes;
	
	ByteLength = BitLength >> 3; //BitLength / 8
	remainder = BitLength & 7;	//BitLength % 8
	if(remainder != 0)
		ByteLength++;

	int wordSize = sizeof(int) * BIT_LEN2INT_LEN(BitLength);
	Bytes = (unsigned char *)malloc(wordSize);
	memset(Bytes, 0, wordSize);

	*numWords = wordSize>>2;	//output number of words

	int byteIdx = 0;
	unsigned char tmp = 0;
	for(i=0; i<BitLength; i++){
		tmp |= Bits[i];
		if(((i+1)&7) == 0){	//(i+1)%8 == 0
			Bytes[byteIdx++] = tmp;
			tmp = 0;
		}
		else
			tmp <<= 1;
	}
	if(remainder != 0)
		Bytes[byteIdx++] = tmp<<(7-remainder);

	return (int *)Bytes;
}

int NumberOfSetBits(int i){
    i = i - ((i >> 1) & 0x55555555);
    i = (i & 0x33333333) + ((i >> 2) & 0x33333333);
    return ((i + (i >> 4) & 0xF0F0F0F) * 0x1010101) >> 24;
}

void invH_mul(unsigned char*decoded, unsigned char* syndrome, unsigned char* invH){
	//(H^-1) * Synd	
	int numWords;
	int *syndWord = BitToWord(syndrome, codeLength, &numWords);
	int * invHint = (int*)invH;
	
	for(int j=0; j<codeLength; j++){
		int tmp = 0;
		for(int k=0; k<numWords; k++){
			tmp ^= invHint[j*numWords + k] & syndWord[k];
		}
		decoded[j] = NumberOfSetBits(tmp) & 1 ;
	}
	free(syndWord);

	//memset(decoded, 0, codeLength * sizeof(unsigned char));	
	//for(int j=0;j<codeLength;j++)            
	//	for(int k=0;k<codeLength;k++) 						
	//		decoded[j] ^=  inv_H_matrix[j*codeLength+k] & Syndrome[k];
}
//void invH_mul_OpenMP()

void LdpcaDecoder(float *LR_intrinsic, unsigned char *accumulatedSyndrome,  
				  unsigned char *decoded, int *rate, 
				  unsigned char crc_encoder, float RateLowerBound)
{  
    int m, n, numInc, nCodeStart, numBytes;
    int code, k, currIndex, prevIndex;    
	unsigned char *decoded_Bytes, crc_decoder;	
	int requestNum = 0;  //start from 0
        
	//syndrome = (unsigned char *)malloc(sizeof(unsigned char) * codeLength); 	

	nCodeStart = 0;
	while (codeRate[nCodeStart] < RateLowerBound) 
		nCodeStart++;

	//compute LLR for beliefPropagation
	float *LLR_intrinsic;
	if(codingOption.ParallelMode==0||codingOption.ParallelMode==1){
		LLR_intrinsic = LR_intrinsic;
		for(k=0;k<codeLength;k++)
			LR_intrinsic[k] = log(LR_intrinsic[k]);
	}

    //iterate through codes of increasing rate
	#ifdef DYNAMICSTEP
	int request = 0;
	int stepInc = 0;
	#endif
    for(code=nCodeStart; code<numCodes; code++, requestNum++)
    {
        m = syndLength[code];
        n = codeLength;
		numInc = bitsPerPeriod[code];

		//printf("code = %d, m = %d, n = %d, numInc = %d\n", code, m, n, numInc);
        rate[0] = m+8;	//symdrome + CRC
		//requestNumber[0] = code - nCodeStart;
        
        currIndex = idxPerPeriod[code][0];
        Syndrome[0] = accumulatedSyndrome[currIndex];
        
        for(k=1; k<m; k++)
        {
            prevIndex = currIndex;
			          // 32 48 65 32 48 65 32...       
            currIndex = idxPerPeriod[code][k%numInc] + (k/numInc)*period; 			
            Syndrome[k] = accumulatedSyndrome[currIndex] ^ accumulatedSyndrome[prevIndex];
        }

		if(rate[0]-8==codeLength)	//use inverse H matrix to recover source
		{
			#ifdef LDPC_ANALYSIS
				ldpcInfo[code].count++;
				TimeStamp start_t, end_t;		
				timeStart(&start_t);
			#endif
			
			//Syndrome * inv_H_matrix
			invH_mul(decoded, Syndrome, inv_H_matrix);	
				
			#ifdef LDPC_ANALYSIS
				double t = timeElapse(start_t,&end_t);
				printf("Inverse H Matrix Multiplication Time: %.2f\n",t);
				ldpcInfo[code].InverseMatrix_time += t;				
			#else
				printf("Inverse H Matrix Multiplication\n");
			#endif
			//free(syndrome);
			
            return;
        }

		int bpResult = 0;
		switch(codingOption.ParallelMode){
			case 0:	//sequential
				bpResult = beliefPropagation(LLR_intrinsic, Syndrome, decoded, m, code);
				break;
			case 1:	//OpenMP only
				bpResult = beliefPropagation_OpenMP(LLR_intrinsic, Syndrome, decoded, m, code);
				break;
			#ifdef CUDA
			case 2: //CUDA only
			case 3: //CUDA + OpenMP
				//bpResult = beliefPropagation(LR_intrinsic, Syndrome, decoded, m, code);
				//bpResult = beliefPropagation_OpenMP(LR_intrinsic, Syndrome, decoded, m, code);
				//bpResult = minsum_beliefPropagation(LR_intrinsic, Syndrome, decoded, m, code);				
				//bpResult = 0;
				if(cuda_concurrent_copy_execution)
					bpResult = beliefPropagation_CUDA_earlyJump(Syndrome, decoded, code, requestNum);
				else
					bpResult = beliefPropagation_CUDA(Syndrome, decoded, code, requestNum);
				break;
			#endif
			default:
				break;
		}
        if(bpResult)
        {            
            //trans to Bytes before CRC
			decoded_Bytes = BitToByte(decoded, codeLength, &numBytes);
			
			//CRC
			crc_decoder = CRC8(decoded_Bytes, numBytes);
			free(decoded_Bytes);

			//check decoder算出來的crc和encoder算的一不一樣
			//一樣的話就表示解對了，不一樣的話就繼續request
			if(crc_decoder == crc_encoder)
			{	
				//free(syndrome);
				return;
			}
        }

		//dynamic step size increament
		#ifdef DYNAMICSTEP
		if(request < JUMPLEN)
			stepInc = jump_step[request];
		code += stepInc;
		if(code >= numCodes-1)
			code = numCodes-2;
		request++;
		#endif		
    }

}
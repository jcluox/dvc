#ifndef LDPCADECODER_H
#define LDPCADECODER_H

#define MAX_ITERATIONS 50

extern unsigned char *Syndrome;
extern char *Buf, *Check_LLR;
extern float *LLR_extrinsic, *Check_LLR_mag, *LLR_overall;


//修改自小小白
////////////////////////////////////////////////////////////////////
// <BitToByte>
//	input:
//		Bits, BitLength 
//	output:
//		unsigned char*, numBytes
////////////////////////////////////////////////////////////////////
unsigned char* BitToByte(unsigned char *Bits, int BitLength, int *numBytes);

////////////////////////////////////////////////////////////////////
// <ByteToBit>
//	input:
//		Bytes, ByteLength, SourceLength
//	output:
//		unsigned char*
////////////////////////////////////////////////////////////////////
unsigned char* ByteToBit(unsigned char *Bytes, int ByteLength, int SourceLength);

float ComputeCE(float *prob);	//prob = p0/p1
float ComputeCE_OpenMP(float *prob);	//prob = p0/p1

void initLdpcaBuffer();
void freeLdpcaBuffer();

int beliefPropagation_OpenMP(float *LLR_intrinsic, unsigned char *syndrome, unsigned char *decoded, int syndromeLength, int nCode);
int beliefPropagation(float *LLR_intrinsic, unsigned char *syndrome, unsigned char *decoded, int syndromeLength, int nCode);
int fast_beliefPropagation(float *LLR_intrinsic, unsigned char *syndrome, unsigned char *decoded, int syndromeLength, int nCode);

void LdpcaDecoder(float *LLR_intrinsic, unsigned char *accumulatedSyndrome,  
				  unsigned char *decoded, int *rate, 
				  unsigned char crc_encoder, float RateLowerBound);

#endif
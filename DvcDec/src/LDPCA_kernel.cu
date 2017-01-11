#include "LDPCA_kernel.h"
#include <stdio.h>

#define EPS 0.000000001f

texture<Msg, 1, cudaReadModeElementType> tex_message;


__device__ __forceinline__  float 
phi(float LLR, int tid){
	//if(tid&1){
	//	return tex1D(tex_LUT, LUT_IDX(LLR));
	//}
	//else{
	//	return (-__logf( tanhf(LLR*0.5f + EPS ) ));
	//}
	return (-__logf( tanhf(LLR*0.5f + EPS ) ));
}

template <int redType, int Size> __device__ void 
reduction(int rowPos, int rowDeg,volatile int s_check[LDPCA_BLOCK_SIZE + 16]);

template <int redType, int Size> __device__ void 
reduction(int rowPos, int rowDeg, volatile float s_LLRmag[LDPCA_BLOCK_SIZE + 16], volatile int s_LLRsign[LDPCA_BLOCK_SIZE + 16]);

template <int redType, int Size> __device__ void 
check_update_reduction(int rowPos, int rowDeg, volatile float s_LLRmag[LDPCA_BLOCK_SIZE + 16], volatile int s_LLRsign_check[LDPCA_BLOCK_SIZE + 16]);


template <int redType, int Size> __global__ void 
InitMessage_kernal(Index2* __restrict__  HinitIdx, char* __restrict__ syndromeByte, float* __restrict__ softInput, 
				   RowInfo* __restrict__ Hinfo,  float* __restrict__ softInputPadding, Msg* __restrict__ InputMessage){
	int tid = blockDim.x*blockIdx.x + threadIdx.x;	
	RowInfo hinfo = Hinfo[tid];
	Index2 idx = HinitIdx[tid];  //GLOBAL read	
    __shared__ int s_LLRsign[LDPCA_BLOCK_SIZE + 16];  //char
    __shared__ float s_LLRmag[LDPCA_BLOCK_SIZE + 16];
	s_LLRsign[threadIdx.x] = 0;
	s_LLRmag[threadIdx.x] = 0.0f;
	if(hinfo.rowDeg){	
		int sign = syndromeByte[idx.syndIdx>>3] ;
		//compute LLR for beliefPropagation
		float LLR = __logf(softInput[idx.softIdx]);  //GLOBAL		

		int rowPos; 
		int rowDeg;
		rowDeg = hinfo.rowDeg;// nextPow2(rowDeg)
		//int half = rowDeg>>1;
		rowPos = threadIdx.x & (rowDeg-1);
		//rowPos = threadIdx.x % rowDeg;  //debug
		softInputPadding[tid] = LLR;  //GLOBAL write	independent instruction

		s_LLRsign[threadIdx.x] = __signbitf(LLR);
		
		sign = (sign >> (7-(idx.syndIdx&0x07))) & 1;
		Hinfo[tid].syndrome = sign;	//GLOBAL write 	independent instruction				
		
		LLR = phi(fabsf(LLR), tid);
		sign = s_LLRsign[threadIdx.x] ^ sign;  //my sign & check node syndrome
		int base = threadIdx.x - rowPos;
		s_LLRmag[threadIdx.x] = LLR;			

		/* Reduction : Compeletely Unroll for each Ladder size*/
		reduction<redType, Size>(rowPos, rowDeg, s_LLRmag, s_LLRsign);

		LLR = phi(s_LLRmag[base] - LLR, tid);
		if(s_LLRsign[base] ^ sign)
			LLR = -LLR;
		#ifdef LDPCA_HALF_FLOAT
			InputMessage[tid] = __float2half_rn(LLR);
		#else
			InputMessage[tid] = LLR;  //GLOBAL write
		#endif
	}
	return;
}
template <int redType, int Size> __global__ void 
#ifndef LDPCA_SINGLE_BUFFER
UpdateMessage_kernal(Msg* __restrict__ OutputMessage, Msg* __restrict__ InputMessage, float* __restrict__ softInputPadding, ColInfo* __restrict__ Vinfo, RowInfo* __restrict__ Hinfo){
#else
UpdateMessage_kernal(Msg* __restrict__ InputMessage, float* __restrict__ softInputPadding, ColInfo* __restrict__ Vinfo, RowInfo* __restrict__ Hinfo){
#endif
	int tid = blockDim.x*blockIdx.x + threadIdx.x;
	ColInfo vinfo = Vinfo[tid];  //GLOBAL    //colNext1 -- colNext2
	RowInfo hinfo = Hinfo[tid];  //GLOBAL    //syndrome -- rowDeg
	__shared__ int s_LLRsign[ LDPCA_BLOCK_SIZE + 16 ];  //char is optimized for fermi
	__shared__ float s_LLRmag[ LDPCA_BLOCK_SIZE + 16 ];
	s_LLRsign[threadIdx.x] = 0;
	s_LLRmag[threadIdx.x] = 0.0f;
	//float soft = softInputPadding[tid];
	if(vinfo.next1!=0||vinfo.next2!=0){
		//========Vertical Processing=========
		#ifdef LDPCA_SINGLE_BUFFER
			#ifdef LDPCA_HALF_FLOAT
				//float LLR = softInputPadding[tid] + __half2float(InputMessage[vinfo.next1]) + __half2float(InputMessage[vinfo.next2]);  //3 GLOBAL read
				float LLR = softInputPadding[tid] + __half2float(tex1Dfetch(tex_message, vinfo.next1)) + __half2float(tex1Dfetch(tex_message, vinfo.next2));  //2texture read ,1 GLOBAL read				
			#else
				float LLR = softInputPadding[tid] + tex1Dfetch(tex_message, vinfo.next1) + tex1Dfetch(tex_message, vinfo.next2);  //2texture read ,1 GLOBAL read		
			#endif
		#else
			#ifdef LDPCA_HALF_FLOAT
				float LLR = softInputPadding[tid] + __half2float(InputMessage[vinfo.next1]) + __half2float(InputMessage[vinfo.next2]);  //3 GLOBAL read
			#else
				float LLR = softInputPadding[tid] + InputMessage[vinfo.next1] + InputMessage[vinfo.next2];  //3 GLOBAL read
			#endif			
		#endif
    //if(tid==0&&redType==0){
        //printf("%d %d %f %d %d %f %f\n",blockDim.x, gridDim.x, softInputPadding[9], Vinfo[11].next2, Hinfo[3].rowDeg, InputMessage[50], LLR);
    //}
		
		//========Horizontal Processing=========
		int rowDeg = hinfo.rowDeg;		
		int sign = hinfo.syndrome ;
		int rowPos = threadIdx.x & (rowDeg-1);

		s_LLRsign[threadIdx.x] = __signbitf(LLR);
		if(LLR<0)
			LLR = -LLR;
		LLR = phi(LLR, tid);
		

		sign = s_LLRsign[threadIdx.x] ^ sign;  //my sign & check node syndrome
		int base = threadIdx.x-rowPos;
		s_LLRmag[threadIdx.x] = LLR;		
		

		/* Reduction : Compeletely Unroll for each Ladder size*/
		reduction<redType, Size>(rowPos, rowDeg, s_LLRmag, s_LLRsign);

		LLR = phi(s_LLRmag[base] - LLR, tid);
		if(s_LLRsign[base] ^ sign)
			LLR = -LLR;

		#ifndef LDPCA_SINGLE_BUFFER
			#ifdef LDPCA_HALF_FLOAT
				OutputMessage[tid] = __float2half_rn(LLR);
			#else
				OutputMessage[tid] = LLR;  //GLOBAL write	
			#endif
		#else
			#ifdef LDPCA_HALF_FLOAT
				InputMessage[tid] = __float2half_rn(LLR);
			#else
				InputMessage[tid] = LLR;  //GLOBAL write	
			#endif
		#endif
	}
	return;
}

template <int redType, int Size> __global__ void 
#ifndef LDPCA_SINGLE_BUFFER
CheckMessage_kernal(int decodedHeaderIdx, Msg* __restrict__ InputMessage, const float* __restrict__ softInputPadding, ColInfo* __restrict__ Vinfo, RowInfo* __restrict__ Hinfo, 
							  unsigned char* __restrict__ decodedInfo){
#else
CheckMessage_kernal(int decodedHeaderIdx, const float* __restrict__ softInputPadding, ColInfo* __restrict__ Vinfo, RowInfo* __restrict__ Hinfo, 
							  unsigned char* __restrict__ decodedInfo){
#endif
	int tid = blockDim.x*blockIdx.x + threadIdx.x;	
	ColInfo vinfo = Vinfo[tid];  //GLOBAL     //colNext1 -- colNext2
	RowInfo hinfo = Hinfo[tid];
	__shared__ int s_check[LDPCA_BLOCK_SIZE + 16];
	s_check[threadIdx.x] = 0;
	int earlyJumpIdx = decodedHeaderIdx;
	if(vinfo.next1!=0||vinfo.next2!=0){
		//========Vertical Processing=========	
		#ifdef LDPCA_SINGLE_BUFFER
			#ifdef LDPCA_HALF_FLOAT
				float mag = softInputPadding[tid] + __half2float(tex1Dfetch(tex_message, tid)) + __half2float(tex1Dfetch(tex_message, vinfo.next1)) + __half2float(tex1Dfetch(tex_message, vinfo.next2));  //2texture read ,1 GLOBAL read				
			#else
				float mag = softInputPadding[tid] + tex1Dfetch(tex_message, tid) + tex1Dfetch(tex_message, vinfo.next1) + tex1Dfetch(tex_message, vinfo.next2);  //4 GLOBAL read
			#endif			
		#else
			#ifdef LDPCA_HALF_FLOAT
				float mag = softInputPadding[tid] + __half2float(InputMessage[tid]) + __half2float(InputMessage[vinfo.next1]) + __half2float(InputMessage[vinfo.next2]);  //3 GLOBAL read
			#else
				float mag = softInputPadding[tid] + InputMessage[tid] + InputMessage[vinfo.next1] + InputMessage[vinfo.next2];  //4 GLOBAL read
			#endif	
			
		#endif
		int rowDeg = hinfo.rowDeg;// nextPow2(rowDeg)
		int rowPos = threadIdx.x & (rowDeg-1);
		//int rowPos = threadIdx.x % rowDeg; //debug
		int decodedBit;
		 //hard decision		
		decodedBit = __signbitf(mag);			

		decodedInfo[hinfo.colIdx] = decodedBit;//GLOBAL				

		s_check[threadIdx.x] = decodedBit;  

		int base = threadIdx.x-rowPos;
		/* Reduction : Compeletely Unroll for each Ladder size*/
		reduction<redType, Size>(rowPos, rowDeg, s_check);
		if(hinfo.syndrome != s_check[base]){
			decodedInfo[earlyJumpIdx]=FAILED_INFO;			//[0] set FAILED_INFO => decoded failed
		}
	}
	return;
}



template <int redType, int Size> __global__ void 
#ifndef LDPCA_SINGLE_BUFFER
CheckUpdateMessage_kernal(int decodedHeaderIdx, Msg* __restrict__ OutputMessage, Msg* __restrict__ InputMessage,  float* __restrict__ softInputPadding, ColInfo* __restrict__ Vinfo, RowInfo* __restrict__ Hinfo, 
						                unsigned char* __restrict__ decodedInfo){
#else
CheckUpdateMessage_kernal(int decodedHeaderIdx, Msg* __restrict__ InputMessage, float* __restrict__ softInputPadding, ColInfo* __restrict__ Vinfo, RowInfo* __restrict__ Hinfo, 
						                unsigned char* __restrict__ decodedInfo){
#endif
	int earlyJumpIdx = decodedHeaderIdx;
	int tid = blockDim.x*blockIdx.x + threadIdx.x;	

	ColInfo vinfo = Vinfo[tid];  //GLOBAL    //colNext1 -- colNext2
	RowInfo hinfo = Hinfo[tid];  //GLOBAL    //syndrome -- rowDeg
	__shared__ int s_LLRsign_check[ LDPCA_BLOCK_SIZE + 16 ];  //char is optimized for fermi
	__shared__ float s_LLRmag[ LDPCA_BLOCK_SIZE + 16 ];
	s_LLRsign_check[threadIdx.x] = 0;
	s_LLRmag[threadIdx.x] = 0.0f;	
	if(tid==0)
		decodedInfo[earlyJumpIdx^INFO_SWAP] = JUMP_INFO;  //global write
	if(vinfo.next1!=0||vinfo.next2!=0){
		//========Vertical Processing=========
		#ifdef LDPCA_SINGLE_BUFFER
			#ifdef LDPCA_HALF_FLOAT
				float LLR = softInputPadding[tid] + __half2float(tex1Dfetch(tex_message, vinfo.next1)) + __half2float(tex1Dfetch(tex_message, vinfo.next2));  //2texture read ,1 GLOBAL read				
				float mag = __half2float(tex1Dfetch(tex_message, tid));
			#else
				float LLR = softInputPadding[tid] + tex1Dfetch(tex_message, vinfo.next1) + tex1Dfetch(tex_message, vinfo.next2);  //2texture read ,1 GLOBAL read		
				float mag = tex1Dfetch(tex_message, tid);
			#endif
		#else
			#ifdef LDPCA_HALF_FLOAT
				float LLR = softInputPadding[tid] + __half2float(InputMessage[vinfo.next1]) + __half2float(InputMessage[vinfo.next2]);  //3 GLOBAL read
				float mag = __half2float(InputMessage[tid]);
			#else
				float LLR = softInputPadding[tid] + InputMessage[vinfo.next1] + InputMessage[vinfo.next2];  //3 GLOBAL read
				float mag =  InputMessage[tid];
			#endif			
		#endif
		//========Horizontal Processing=========
		int  decodedBit = __signbitf(mag + LLR);
		decodedInfo[hinfo.colIdx] = decodedBit;  //global
		s_LLRsign_check[threadIdx.x] = __signbitf(LLR) | (decodedBit<<1);  //store both sign bit and decoded bit (their reduction operations are the same)
		
		int rowDeg = hinfo.rowDeg;		
		int rowPos = threadIdx.x & (rowDeg-1);

		if(LLR<0)
			LLR = -LLR;
		LLR = phi(LLR, tid);
		

		int sign = s_LLRsign_check[threadIdx.x] ^ hinfo.syndrome;  //my sign & check node syndrome

		int base = threadIdx.x-rowPos;
		s_LLRmag[threadIdx.x] = LLR;
		

		/* Check Reduction : Compeletely Unroll for each Ladder size*/
		/* Update Reduction : Compeletely Unroll for each Ladder size*/
		check_update_reduction<redType, Size>(rowPos, rowDeg, s_LLRmag, s_LLRsign_check);
		//reduction<redType, Size>(rowPos, rowDeg, s_LLRmag, s_LLRsign_check);
		//reduction<redType, Size>(rowPos, rowDeg, s_check);
		

		LLR = phi(s_LLRmag[base] - LLR, tid);
		if(((s_LLRsign_check[base] ^ sign)&1))
			LLR = -LLR;

		#ifndef LDPCA_SINGLE_BUFFER
			#ifdef LDPCA_HALF_FLOAT
				OutputMessage[tid] = __float2half_rn(LLR);
			#else
				OutputMessage[tid] = LLR;  //GLOBAL write	
			#endif
		#else
			#ifdef LDPCA_HALF_FLOAT
				InputMessage[tid] = __float2half_rn(LLR);
			#else
				InputMessage[tid] = LLR;  //GLOBAL write	
			#endif
		#endif

		if(hinfo.syndrome != (s_LLRsign_check[base]>>1)){
			decodedInfo[earlyJumpIdx]=FAILED_INFO;			//[0] set FAILED_INFO => decoded failed (NO positive early jump)
		}

	}
	return;
}

template <int redType, int Size> __device__ void 
check_update_reduction(int rowPos, int rowDeg, volatile float s_LLRmag[LDPCA_BLOCK_SIZE + 16], volatile int s_LLRsign_check[LDPCA_BLOCK_SIZE + 16]){
	/* Reduction : Compeletely Unroll for each Ladder size*/
	switch (redType){
		case 0:  //rowSum 99x48  //CIF rowSum 98x15, 99x162
			__syncthreads();
			if (rowPos <  64){
				s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 64];  
				s_LLRsign_check[threadIdx.x] ^= s_LLRsign_check[threadIdx.x + 64];					
			}
			__syncthreads();
			if (rowPos <  32){
				s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 32];  
				s_LLRsign_check[threadIdx.x] ^= s_LLRsign_check[threadIdx.x + 32];
				s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 16];  
				s_LLRsign_check[threadIdx.x] ^= s_LLRsign_check[threadIdx.x + 16];
				s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 8];  
				s_LLRsign_check[threadIdx.x] ^= s_LLRsign_check[threadIdx.x + 8];
				s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 4];  
				s_LLRsign_check[threadIdx.x] ^= s_LLRsign_check[threadIdx.x + 4];
				s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 2];  
				s_LLRsign_check[threadIdx.x] ^= s_LLRsign_check[threadIdx.x + 2];
				s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 1];  
				s_LLRsign_check[threadIdx.x] ^= s_LLRsign_check[threadIdx.x + 1];
			}
			__syncthreads();
			break;
		case 1:  //rowSum 48x24, 51x24, 99x24  //CIF rowSum 48x96, 50x5, 51x84, 52x7, 98x10, 99x78
			__syncthreads();
			if(rowDeg > 64 ){  //99
				if( rowPos < 64){
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 64];  
					s_LLRsign_check[threadIdx.x] ^= s_LLRsign_check[threadIdx.x + 64];	
				}
				__syncthreads();
			}
			if (rowPos < 32){
				s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 32];  
				s_LLRsign_check[threadIdx.x] ^= s_LLRsign_check[threadIdx.x + 32];
				s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 16];  
				s_LLRsign_check[threadIdx.x] ^= s_LLRsign_check[threadIdx.x + 16];
				s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 8];  
				s_LLRsign_check[threadIdx.x] ^= s_LLRsign_check[threadIdx.x + 8];
				s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 4];  
				s_LLRsign_check[threadIdx.x] ^= s_LLRsign_check[threadIdx.x + 4];
				s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 2];  
				s_LLRsign_check[threadIdx.x] ^= s_LLRsign_check[threadIdx.x + 2];
				s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 1];  
				s_LLRsign_check[threadIdx.x] ^= s_LLRsign_check[threadIdx.x + 1];
			}
			__syncthreads();
			break;
		case 2:   //rowSum 48x48  51x48  //CIF rowSum 48x192, 50x15, 51x162, 52x15
			__syncthreads();
			if (rowPos < 32){
				s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 32];  
				s_LLRsign_check[threadIdx.x] ^= s_LLRsign_check[threadIdx.x + 32];
				s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 16];  
				s_LLRsign_check[threadIdx.x] ^= s_LLRsign_check[threadIdx.x + 16];
				s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 8];  
				s_LLRsign_check[threadIdx.x] ^= s_LLRsign_check[threadIdx.x + 8];
				s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 4];  
				s_LLRsign_check[threadIdx.x] ^= s_LLRsign_check[threadIdx.x + 4];
				s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 2];  
				s_LLRsign_check[threadIdx.x] ^= s_LLRsign_check[threadIdx.x + 2];
				s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 1];  
				s_LLRsign_check[threadIdx.x] ^= s_LLRsign_check[threadIdx.x + 1];
			}
			__syncthreads();
			break;
		case 3:   //rowSum 24x24  27x24  48x48  51x24  //CIF rowSum 23x5, 24x84, 25x7, 27x96, 48x192, 50x10, 51x78, 52x8
		case 4:   //rowSum 24x48  27x48  48x48  //CIF rowSum 23x15, 24x162, 25x15, 27x192, 48x192
		case 5:   //rowSum 24x96  27x48  48x24  //CIF rowSum 23x15, 24x354, 25x15, 27x192, 48x96
			if(rowDeg <= 32 ){  //24  27
				s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 16];  
				s_LLRsign_check[threadIdx.x] ^= s_LLRsign_check[threadIdx.x + 16];
				s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 8];  
				s_LLRsign_check[threadIdx.x] ^= s_LLRsign_check[threadIdx.x + 8];
				s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 4];  
				s_LLRsign_check[threadIdx.x] ^= s_LLRsign_check[threadIdx.x + 4];
				s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 2];  
				s_LLRsign_check[threadIdx.x] ^= s_LLRsign_check[threadIdx.x + 2];
				s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 1];  
				s_LLRsign_check[threadIdx.x] ^= s_LLRsign_check[threadIdx.x + 1];
			}
			else{  //48 51
				__syncthreads();
				if (rowPos < 32){
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 32];  
					s_LLRsign_check[threadIdx.x] ^= s_LLRsign_check[threadIdx.x + 32];
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 16];  
					s_LLRsign_check[threadIdx.x] ^= s_LLRsign_check[threadIdx.x + 16];
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 8];  
					s_LLRsign_check[threadIdx.x] ^= s_LLRsign_check[threadIdx.x + 8];
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 4];  
					s_LLRsign_check[threadIdx.x] ^= s_LLRsign_check[threadIdx.x + 4];
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 2];  
					s_LLRsign_check[threadIdx.x] ^= s_LLRsign_check[threadIdx.x + 2];
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 1];  
					s_LLRsign_check[threadIdx.x] ^= s_LLRsign_check[threadIdx.x + 1];
				}
				__syncthreads();
			}
			break;
		case 6:   //rowSum 24x144  27x48  //CIF rowSum 23x15, 24x546, 25x15, 27x192
			s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 16];  
			s_LLRsign_check[threadIdx.x] ^= s_LLRsign_check[threadIdx.x + 16];
			s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 8];  
			s_LLRsign_check[threadIdx.x] ^= s_LLRsign_check[threadIdx.x + 8];
			s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 4];  
			s_LLRsign_check[threadIdx.x] ^= s_LLRsign_check[threadIdx.x + 4];
			s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 2];  
			s_LLRsign_check[threadIdx.x] ^= s_LLRsign_check[threadIdx.x + 2];
			s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 1];  
			s_LLRsign_check[threadIdx.x] ^= s_LLRsign_check[threadIdx.x + 1];
			break;
		case 7:    //12x24  15x24  24x144  27x24  //CIF rowSum 12x96, 15x96, 23x15, 24x546, 25x15, 27x96
		case 8:    //12x48  15x48  24x144  //CIF rowSum 12x192, 15x192, 23x15, 24x546, 25x15
		case 9:    //12x96  15x48  24x120  //CIF rowSum 12x384, 15x192, 23x15, 24x450, 25x15
		case 10:   //12x144 15x48  24x96  //CIF rowSum 12x576, 15x192, 23x15, 24x354, 25x15
		case 11:   //12x192 15x48  24x72  //CIF rowSum 11x5, 12x756, 13x7, 15x192, 23x10, 24x270, 25x8
		case 12:   //12x240 15x48  24x48  //CIF rowSum 11x15, 12x930, 13x15, 15x192, 24x192
		case 13:   //12x288 15x48  24x24  //CIF rowSum 11x15, 12x1122, 13x15, 15x192, 24x96
			if(rowDeg <= 16 ){  //12  15
				s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 8];  
				s_LLRsign_check[threadIdx.x] ^= s_LLRsign_check[threadIdx.x + 8];
				s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 4];  
				s_LLRsign_check[threadIdx.x] ^= s_LLRsign_check[threadIdx.x + 4];
				s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 2];  
				s_LLRsign_check[threadIdx.x] ^= s_LLRsign_check[threadIdx.x + 2];
				s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 1];  
				s_LLRsign_check[threadIdx.x] ^= s_LLRsign_check[threadIdx.x + 1];
			}
			else{  //24 27
				s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 16];  
				s_LLRsign_check[threadIdx.x] ^= s_LLRsign_check[threadIdx.x + 16];
				s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 8];  
				s_LLRsign_check[threadIdx.x] ^= s_LLRsign_check[threadIdx.x + 8];
				s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 4];  
				s_LLRsign_check[threadIdx.x] ^= s_LLRsign_check[threadIdx.x + 4];
				s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 2];  
				s_LLRsign_check[threadIdx.x] ^= s_LLRsign_check[threadIdx.x + 2];
				s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 1];  
				s_LLRsign_check[threadIdx.x] ^= s_LLRsign_check[threadIdx.x + 1];
			}
			break;
		case 14:    //12x336  15x48  //CIF rowSum 11x15, 12x1314, 13x15, 15x192
			s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 8];  
			s_LLRsign_check[threadIdx.x] ^= s_LLRsign_check[threadIdx.x + 8];
			s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 4];  
			s_LLRsign_check[threadIdx.x] ^= s_LLRsign_check[threadIdx.x + 4];
			s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 2];  
			s_LLRsign_check[threadIdx.x] ^= s_LLRsign_check[threadIdx.x + 2];
			s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 1];  
			s_LLRsign_check[threadIdx.x] ^= s_LLRsign_check[threadIdx.x + 1];
			break;
		case 15:    //6x24  9x24  12x336  15x24  //CIF rowSum 6x96, 9x96, 11x15, 12x1314, 13x15, 15x96
		case 16:    //6x48  9x48  12x336  //CIF rowSum 6x192, 9x192, 11x15, 12x1314, 13x15
		case 17:    //6x96  9x48  12x312  //CIF rowSum 6x384, 9x192, 11x15, 12x1218, 13x15
		case 18:    //6x144  9x48  12x288  //CIF rowSum 6x576, 9x192, 11x15, 12x1122, 13x15
		case 19:    //6x192  9x48  12x264  //CIF rowSum 6x768, 9x192, 11x15, 12x1026, 13x15
		case 20:    //6x240  9x48  12x240  //CIF rowSum 6x960, 9x192, 11x15, 12x930, 13x15
		case 21:    //6x288  9x48  12x216  //CIF rowSum 6x1152, 9x192, 11x15, 12x834, 13x15
		case 22:    //6x336  9x48  12x192  //CIF rowSum 6x1344, 9x192, 11x15, 12x738, 13x15
		case 23:    //6x384  9x48  12x168  //CIF rowSum 6x1536, 9x192, 11x15, 12x642, 13x15
		case 24:    //6x432  9x48  12x144  //CIF rowSum 6x1728, 9x192, 11x15, 12x546, 13x15
		case 25:    //6x480  9x48  12x120  //CIF rowSum 6x1920, 9x192, 11x15, 12x450, 13x15
		case 26:    //6x528  9x48  12x96  //CIF rowSum 6x2112, 9x192, 11x15, 12x354, 13x15
		case 27:    //6x576  9x48  12x72  //CIF rowSum 5x5, 6x2292, 7x7, 9x192, 11x10, 12x270, 13x8
		case 28:    //6x624  9x48  12x48  //CIF rowSum 5x15, 6x2466, 7x15, 9x192, 12x192
		case 29:    //6x672  9x48  12x24  //CIF rowSum 5x15, 6x2658, 7x15, 9x192, 12x96
		case 30:    //6x720  9x48         //CIF rowSum 5x15, 6x2850, 7x15, 9x192
			if(rowDeg <= 8 ){  //6
				s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 4];  
				s_LLRsign_check[threadIdx.x] ^= s_LLRsign_check[threadIdx.x + 4];
				s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 2];  
				s_LLRsign_check[threadIdx.x] ^= s_LLRsign_check[threadIdx.x + 2];
				s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 1];  
				s_LLRsign_check[threadIdx.x] ^= s_LLRsign_check[threadIdx.x + 1];
			}
			else{  //9 12 15
				s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 8];  
				s_LLRsign_check[threadIdx.x] ^= s_LLRsign_check[threadIdx.x + 8];
				s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 4];  
				s_LLRsign_check[threadIdx.x] ^= s_LLRsign_check[threadIdx.x + 4];
				s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 2];  
				s_LLRsign_check[threadIdx.x] ^= s_LLRsign_check[threadIdx.x + 2];
				s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 1];  
				s_LLRsign_check[threadIdx.x] ^= s_LLRsign_check[threadIdx.x + 1];
			}
			break;
		case 31:    //3x24  6x744  9x24  //CIF rowSum 3x96, 5x15, 6x2946, 7x15, 9x96
			if(rowDeg <= 4 ){  //3  
				s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 2];  
				s_LLRsign_check[threadIdx.x] ^= s_LLRsign_check[threadIdx.x + 2];
				s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 1];  
				s_LLRsign_check[threadIdx.x] ^= s_LLRsign_check[threadIdx.x + 1];
			}
			else if(rowDeg <= 8){  //6
				s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 4];  
				s_LLRsign_check[threadIdx.x] ^= s_LLRsign_check[threadIdx.x + 4];
				s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 2];  
				s_LLRsign_check[threadIdx.x] ^= s_LLRsign_check[threadIdx.x + 2];
				s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 1];  
				s_LLRsign_check[threadIdx.x] ^= s_LLRsign_check[threadIdx.x + 1];
			}
			else{  //9
				s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 8];  
				s_LLRsign_check[threadIdx.x] ^= s_LLRsign_check[threadIdx.x + 8];
				s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 4];  
				s_LLRsign_check[threadIdx.x] ^= s_LLRsign_check[threadIdx.x + 4];
				s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 2];  
				s_LLRsign_check[threadIdx.x] ^= s_LLRsign_check[threadIdx.x + 2];
				s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 1];  
				s_LLRsign_check[threadIdx.x] ^= s_LLRsign_check[threadIdx.x + 1];
			}
			break;
		case 32:    //3x48  6x768  //CIF rowSum 3x96, 5x15, 6x2946, 7x15, 9x96
			if(Size==CIF){
				if(rowDeg <= 4 ){  //3  
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 2];  
					s_LLRsign_check[threadIdx.x] ^= s_LLRsign_check[threadIdx.x + 2];
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 1];  
					s_LLRsign_check[threadIdx.x] ^= s_LLRsign_check[threadIdx.x + 1];
				}
				else if(rowDeg <= 8){  //6
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 4];  
					s_LLRsign_check[threadIdx.x] ^= s_LLRsign_check[threadIdx.x + 4];
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 2];  
					s_LLRsign_check[threadIdx.x] ^= s_LLRsign_check[threadIdx.x + 2];
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 1];  
					s_LLRsign_check[threadIdx.x] ^= s_LLRsign_check[threadIdx.x + 1];
				}
				else{  //9
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 8];  
					s_LLRsign_check[threadIdx.x] ^= s_LLRsign_check[threadIdx.x + 8];
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 4];  
					s_LLRsign_check[threadIdx.x] ^= s_LLRsign_check[threadIdx.x + 4];
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 2];  
					s_LLRsign_check[threadIdx.x] ^= s_LLRsign_check[threadIdx.x + 2];
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 1];  
					s_LLRsign_check[threadIdx.x] ^= s_LLRsign_check[threadIdx.x + 1];
				}
				break;
			}
		case 33:    //3x96  6x744  //CIF rowSum 3x192, 5x15, 6x3042, 7x15
		case 34:    //3x144  6x720	  //CIF rowSum 3x384, 5x15, 6x2946, 7x15
		case 35:    //3x192  6x696	  //CIF rowSum 3x576, 5x15, 6x2850, 7x15			
		case 36:    //3x240  6x672   //CIF rowSum 3x960, 5x15, 6x2658, 7x15
		case 37:    //3x288  6x648	//CIF rowSum 3x1152, 5x15, 6x2562, 7x15		
		case 38:    //3x336  6x624	//CIF rowSum 3x1344, 5x15, 6x2466, 7x15	
		case 39:    //3x384  6x600	//CIF rowSum 3x1536, 5x15, 6x2370, 7x15		
		case 40:    //3x432  6x576	//CIF rowSum 3x1728, 5x15, 6x2274, 7x15			
		case 41:    //3x480  6x552	//CIF rowSum 3x1920, 5x15, 6x2178, 7x15			
		case 42:    //3x528  6x528	//CIF rowSum 3x2112, 5x15, 6x2082, 7x15			
		case 43:    //3x576  6x504	//CIF rowSum 3x2304, 5x15, 6x1986, 7x15			
		case 44:    //3x624  6x480	//CIF rowSum 3x2496, 5x15, 6x1890, 7x15			
		case 45:    //3x672  6x456	//CIF rowSum 3x2688, 5x15, 6x1794, 7x15			
		case 46:    //3x720  6x432	//CIF rowSum 3x2880, 5x15, 6x1698, 7x15			
		case 47:    //3x768  6x408	//CIF rowSum 3x3072, 5x15, 6x1602, 7x15			
		case 48:    //3x816  6x384	//CIF rowSum 3x3264, 5x15, 6x1506, 7x15			
		case 49:    //3x864  6x360	//CIF rowSum 3x3456, 5x15, 6x1410, 7x15			
		case 50:    //3x912  6x336	//CIF rowSum 3x3648, 5x15, 6x1314, 7x15			
		case 51:    //3x960  6x312	//CIF rowSum 3x3840, 5x15, 6x1218, 7x15			
		case 52:    //3x1008  6x288	//CIF rowSum 3x4032, 5x15, 6x1122, 7x15			
		case 53:    //3x1056  6x264	//CIF rowSum 3x4224, 5x15, 6x1026, 7x15			
		case 54:    //3x1104  6x240	//CIF rowSum 3x4416, 5x15, 6x930, 7x15			
		case 55:    //3x1152  6x216	//CIF rowSum 3x4608, 5x15, 6x834, 7x15			
		case 56:    //3x1200  6x192	//CIF rowSum 3x4800, 5x15, 6x738, 7x15			
		case 57:    //3x1248  6x168	 //CIF rowSum 3x4992, 5x15, 6x642, 7x15			
		case 58:    //3x1296  6x144	//CIF rowSum 3x5184, 5x15, 6x546, 7x15			
			if(rowDeg <= 4 ){  //3  
				s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 2];  
				s_LLRsign_check[threadIdx.x] ^= s_LLRsign_check[threadIdx.x + 2];
				s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 1];  
				s_LLRsign_check[threadIdx.x] ^= s_LLRsign_check[threadIdx.x + 1];
			}
			else{  //6
				s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 4];  
				s_LLRsign_check[threadIdx.x] ^= s_LLRsign_check[threadIdx.x + 4];
				s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 2];  
				s_LLRsign_check[threadIdx.x] ^= s_LLRsign_check[threadIdx.x + 2];
				s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 1];  
				s_LLRsign_check[threadIdx.x] ^= s_LLRsign_check[threadIdx.x + 1];
			}
			break;
		case 59:    //3x1344  6x120	//CIF rowSum 2x5, 3x5364, 4x7, 5x10, 6x462, 7x8			
		case 60:    //3x1392  6x96	//CIF rowSum 2x15, 3x5538, 4x15, 6x384			
		case 61:    //3x1440  6x72	//CIF rowSum 2x15, 3x5730, 4x15, 6x288			
		case 62:    //3x1488  6x48	//CIF rowSum 2x15, 3x5922, 4x15, 6x192			
		case 63:    //3x1536  6x24  //CIF rowSum 2x15, 3x6114, 4x15, 6x96
			if(Size==QCIF){
				if(rowDeg <= 4 ){  //3  
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 2];  
					s_LLRsign_check[threadIdx.x] ^= s_LLRsign_check[threadIdx.x + 2];
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 1];  
					s_LLRsign_check[threadIdx.x] ^= s_LLRsign_check[threadIdx.x + 1];
				}
				else{  //6
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 4];  
					s_LLRsign_check[threadIdx.x] ^= s_LLRsign_check[threadIdx.x + 4];
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 2];  
					s_LLRsign_check[threadIdx.x] ^= s_LLRsign_check[threadIdx.x + 2];
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 1];  
					s_LLRsign_check[threadIdx.x] ^= s_LLRsign_check[threadIdx.x + 1];
				}
				break;
			}
			else{
				if(rowDeg == 2){
						s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 1];
						s_LLRsign_check[threadIdx.x] ^= s_LLRsign_check[threadIdx.x + 1];
				}
				else if(rowDeg <= 4 ){  //3  4
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 2];  
					s_LLRsign_check[threadIdx.x] ^= s_LLRsign_check[threadIdx.x + 2];
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 1];  
					s_LLRsign_check[threadIdx.x] ^= s_LLRsign_check[threadIdx.x + 1];
				}
				else{  //5 6 7
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 4];  
					s_LLRsign_check[threadIdx.x] ^= s_LLRsign_check[threadIdx.x + 4];
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 2];  
					s_LLRsign_check[threadIdx.x] ^= s_LLRsign_check[threadIdx.x + 2];
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 1];  
					s_LLRsign_check[threadIdx.x] ^= s_LLRsign_check[threadIdx.x + 1];
				}
				break;
			}			
		/*default:
			__syncthreads();
			do{
				if(rowPos < half){
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x+half];  
					s_LLRsign_check[threadIdx.x] ^= s_LLRsign_check[threadIdx.x+half];
				}
				half >>= 1;
				__syncthreads();
			}while(half);				
			break;*/
	}	
}

template <int redType, int Size> __device__ void 
reduction(int rowPos, int rowDeg, volatile float s_LLRmag[LDPCA_BLOCK_SIZE + 16], volatile int s_LLRsign[LDPCA_BLOCK_SIZE + 16]){
	/* Reduction : Compeletely Unroll for each Ladder size*/
	switch (redType){
		case 0:  //rowSum 99x48  //CIF rowSum 98x15, 99x162
			__syncthreads();
			if (rowPos <  64){
				s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 64];  
				s_LLRsign[threadIdx.x] ^= s_LLRsign[threadIdx.x + 64];					
			}
			__syncthreads();
			if (rowPos <  32){
				s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 32];  
				s_LLRsign[threadIdx.x] ^= s_LLRsign[threadIdx.x + 32];
				#if __CUDA_ARCH__ >= 200
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 16];  
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 8];  
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 4];  
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 2];  
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 1];  
					s_LLRsign[threadIdx.x] = __popc(__ballot(s_LLRsign[threadIdx.x])) & 1; // Magic happens here
				#else
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 16];  
					s_LLRsign[threadIdx.x] ^= s_LLRsign[threadIdx.x + 16];
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 8];  
					s_LLRsign[threadIdx.x] ^= s_LLRsign[threadIdx.x + 8];
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 4];  
					s_LLRsign[threadIdx.x] ^= s_LLRsign[threadIdx.x + 4];
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 2];  
					s_LLRsign[threadIdx.x] ^= s_LLRsign[threadIdx.x + 2];
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 1];  
					s_LLRsign[threadIdx.x] ^= s_LLRsign[threadIdx.x + 1];
				#endif
			}
			__syncthreads();
			break;
		case 1:  //rowSum 48x24, 51x24, 99x24  //CIF rowSum 48x96, 50x5, 51x84, 52x7, 98x10, 99x78
			__syncthreads();
			if(rowDeg > 64 ){  //99
				if( rowPos < 64){
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 64];  
					s_LLRsign[threadIdx.x] ^= s_LLRsign[threadIdx.x + 64];	
				}
				__syncthreads();
			}
			if (rowPos < 32){
				s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 32];  
				s_LLRsign[threadIdx.x] ^= s_LLRsign[threadIdx.x + 32];
				#if __CUDA_ARCH__ >= 200
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 16];  
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 8];  
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 4];  
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 2];  
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 1];  
					s_LLRsign[threadIdx.x] = __popc(__ballot(s_LLRsign[threadIdx.x])) & 1; // Magic happens here
				#else
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 16];  
					s_LLRsign[threadIdx.x] ^= s_LLRsign[threadIdx.x + 16];
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 8];  
					s_LLRsign[threadIdx.x] ^= s_LLRsign[threadIdx.x + 8];
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 4];  
					s_LLRsign[threadIdx.x] ^= s_LLRsign[threadIdx.x + 4];
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 2];  
					s_LLRsign[threadIdx.x] ^= s_LLRsign[threadIdx.x + 2];
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 1];  
					s_LLRsign[threadIdx.x] ^= s_LLRsign[threadIdx.x + 1];
				#endif
			}
			__syncthreads();
			break;
		case 2:   //rowSum 48x48  51x48  //CIF rowSum 48x192, 50x15, 51x162, 52x15
			__syncthreads();
			if (rowPos < 32){
				s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 32];  
				s_LLRsign[threadIdx.x] ^= s_LLRsign[threadIdx.x + 32];
				#if __CUDA_ARCH__ >= 200
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 16];  
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 8];  
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 4];  
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 2];  
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 1];  
					s_LLRsign[threadIdx.x] = __popc(__ballot(s_LLRsign[threadIdx.x])) & 1; // Magic happens here
				#else
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 16];  
					s_LLRsign[threadIdx.x] ^= s_LLRsign[threadIdx.x + 16];
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 8];  
					s_LLRsign[threadIdx.x] ^= s_LLRsign[threadIdx.x + 8];
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 4];  
					s_LLRsign[threadIdx.x] ^= s_LLRsign[threadIdx.x + 4];
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 2];  
					s_LLRsign[threadIdx.x] ^= s_LLRsign[threadIdx.x + 2];
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 1];  
					s_LLRsign[threadIdx.x] ^= s_LLRsign[threadIdx.x + 1];
				#endif
			}
			__syncthreads();
			break;
		case 3:   //rowSum 24x24  27x24  48x48  51x24  //CIF rowSum 23x5, 24x84, 25x7, 27x96, 48x192, 50x10, 51x78, 52x8
		case 4:   //rowSum 24x48  27x48  48x48  //CIF rowSum 23x15, 24x162, 25x15, 27x192, 48x192
		case 5:   //rowSum 24x96  27x48  48x24  //CIF rowSum 23x15, 24x354, 25x15, 27x192, 48x96
			if(rowDeg <= 32 ){  //24  27
				#if __CUDA_ARCH__ >= 200
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 16];  
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 8];  
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 4];  
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 2];  
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 1];  
					s_LLRsign[threadIdx.x] = __popc(__ballot(s_LLRsign[threadIdx.x])) & 1; // Magic happens here
				#else
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 16];  
					s_LLRsign[threadIdx.x] ^= s_LLRsign[threadIdx.x + 16];
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 8];  
					s_LLRsign[threadIdx.x] ^= s_LLRsign[threadIdx.x + 8];
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 4];  
					s_LLRsign[threadIdx.x] ^= s_LLRsign[threadIdx.x + 4];
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 2];  
					s_LLRsign[threadIdx.x] ^= s_LLRsign[threadIdx.x + 2];
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 1];  
					s_LLRsign[threadIdx.x] ^= s_LLRsign[threadIdx.x + 1];
				#endif				
			}
			else{  //48 51
				__syncthreads();
				if (rowPos < 32){
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 32];  
					s_LLRsign[threadIdx.x] ^= s_LLRsign[threadIdx.x + 32];
					#if __CUDA_ARCH__ >= 200
						s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 16];  
						s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 8];  
						s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 4];  
						s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 2];  
						s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 1];  
						s_LLRsign[threadIdx.x] = __popc(__ballot(s_LLRsign[threadIdx.x])) & 1; // Magic happens here
					#else
						s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 16];  
						s_LLRsign[threadIdx.x] ^= s_LLRsign[threadIdx.x + 16];
						s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 8];  
						s_LLRsign[threadIdx.x] ^= s_LLRsign[threadIdx.x + 8];
						s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 4];  
						s_LLRsign[threadIdx.x] ^= s_LLRsign[threadIdx.x + 4];
						s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 2];  
						s_LLRsign[threadIdx.x] ^= s_LLRsign[threadIdx.x + 2];
						s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 1];  
						s_LLRsign[threadIdx.x] ^= s_LLRsign[threadIdx.x + 1];
					#endif
				}
				__syncthreads();
			}
			break;
		case 6:   //rowSum 24x144  27x48  //CIF rowSum 23x15, 24x546, 25x15, 27x192
			#if __CUDA_ARCH__ >= 200
				s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 16];  
				s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 8];  
				s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 4];  
				s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 2];  
				s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 1];  
				s_LLRsign[threadIdx.x] = __popc(__ballot(s_LLRsign[threadIdx.x])) & 1; // Magic happens here
			#else
				s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 16];  
				s_LLRsign[threadIdx.x] ^= s_LLRsign[threadIdx.x + 16];
				s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 8];  
				s_LLRsign[threadIdx.x] ^= s_LLRsign[threadIdx.x + 8];
				s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 4];  
				s_LLRsign[threadIdx.x] ^= s_LLRsign[threadIdx.x + 4];
				s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 2];  
				s_LLRsign[threadIdx.x] ^= s_LLRsign[threadIdx.x + 2];
				s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 1];  
				s_LLRsign[threadIdx.x] ^= s_LLRsign[threadIdx.x + 1];
			#endif			
			break;
		case 7:    //12x24  15x24  24x144  27x24  //CIF rowSum 12x96, 15x96, 23x15, 24x546, 25x15, 27x96
		case 8:    //12x48  15x48  24x144  //CIF rowSum 12x192, 15x192, 23x15, 24x546, 25x15
		case 9:    //12x96  15x48  24x120  //CIF rowSum 12x384, 15x192, 23x15, 24x450, 25x15
		case 10:   //12x144 15x48  24x96  //CIF rowSum 12x576, 15x192, 23x15, 24x354, 25x15
		case 11:   //12x192 15x48  24x72  //CIF rowSum 11x5, 12x756, 13x7, 15x192, 23x10, 24x270, 25x8
		case 12:   //12x240 15x48  24x48  //CIF rowSum 11x15, 12x930, 13x15, 15x192, 24x192
		case 13:   //12x288 15x48  24x24  //CIF rowSum 11x15, 12x1122, 13x15, 15x192, 24x96
			if(rowDeg <= 16 ){  //12  15
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 8];  
					s_LLRsign[threadIdx.x] ^= s_LLRsign[threadIdx.x + 8];
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 4];  
					s_LLRsign[threadIdx.x] ^= s_LLRsign[threadIdx.x + 4];
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 2];  
					s_LLRsign[threadIdx.x] ^= s_LLRsign[threadIdx.x + 2];
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 1];  
					s_LLRsign[threadIdx.x] ^= s_LLRsign[threadIdx.x + 1];
			}
			else{  //24 27
				#if __CUDA_ARCH__ >= 200
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 16];  
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 8];  
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 4];  
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 2];  
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 1];  
					s_LLRsign[threadIdx.x] = __popc(__ballot(s_LLRsign[threadIdx.x])) & 1; // Magic happens here
				#else
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 16];  
					s_LLRsign[threadIdx.x] ^= s_LLRsign[threadIdx.x + 16];
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 8];  
					s_LLRsign[threadIdx.x] ^= s_LLRsign[threadIdx.x + 8];
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 4];  
					s_LLRsign[threadIdx.x] ^= s_LLRsign[threadIdx.x + 4];
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 2];  
					s_LLRsign[threadIdx.x] ^= s_LLRsign[threadIdx.x + 2];
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 1];  
					s_LLRsign[threadIdx.x] ^= s_LLRsign[threadIdx.x + 1];
				#endif				
			}
			break;
		case 14:    //12x336  15x48  //CIF rowSum 11x15, 12x1314, 13x15, 15x192
				s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 8];  
				s_LLRsign[threadIdx.x] ^= s_LLRsign[threadIdx.x + 8];
				s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 4];  
				s_LLRsign[threadIdx.x] ^= s_LLRsign[threadIdx.x + 4];
				s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 2];  
				s_LLRsign[threadIdx.x] ^= s_LLRsign[threadIdx.x + 2];
				s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 1];  
				s_LLRsign[threadIdx.x] ^= s_LLRsign[threadIdx.x + 1];
			break;
		case 15:    //6x24  9x24  12x336  15x24  //CIF rowSum 6x96, 9x96, 11x15, 12x1314, 13x15, 15x96
		case 16:    //6x48  9x48  12x336  //CIF rowSum 6x192, 9x192, 11x15, 12x1314, 13x15
		case 17:    //6x96  9x48  12x312  //CIF rowSum 6x384, 9x192, 11x15, 12x1218, 13x15
		case 18:    //6x144  9x48  12x288  //CIF rowSum 6x576, 9x192, 11x15, 12x1122, 13x15
		case 19:    //6x192  9x48  12x264  //CIF rowSum 6x768, 9x192, 11x15, 12x1026, 13x15
		case 20:    //6x240  9x48  12x240  //CIF rowSum 6x960, 9x192, 11x15, 12x930, 13x15
		case 21:    //6x288  9x48  12x216  //CIF rowSum 6x1152, 9x192, 11x15, 12x834, 13x15
		case 22:    //6x336  9x48  12x192  //CIF rowSum 6x1344, 9x192, 11x15, 12x738, 13x15
		case 23:    //6x384  9x48  12x168  //CIF rowSum 6x1536, 9x192, 11x15, 12x642, 13x15
		case 24:    //6x432  9x48  12x144  //CIF rowSum 6x1728, 9x192, 11x15, 12x546, 13x15
		case 25:    //6x480  9x48  12x120  //CIF rowSum 6x1920, 9x192, 11x15, 12x450, 13x15
		case 26:    //6x528  9x48  12x96  //CIF rowSum 6x2112, 9x192, 11x15, 12x354, 13x15
		case 27:    //6x576  9x48  12x72  //CIF rowSum 5x5, 6x2292, 7x7, 9x192, 11x10, 12x270, 13x8
		case 28:    //6x624  9x48  12x48  //CIF rowSum 5x15, 6x2466, 7x15, 9x192, 12x192
		case 29:    //6x672  9x48  12x24  //CIF rowSum 5x15, 6x2658, 7x15, 9x192, 12x96
		case 30:    //6x720  9x48         //CIF rowSum 5x15, 6x2850, 7x15, 9x192
			if(rowDeg <= 8 ){  //6
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 4];  
					s_LLRsign[threadIdx.x] ^= s_LLRsign[threadIdx.x + 4];
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 2];  
					s_LLRsign[threadIdx.x] ^= s_LLRsign[threadIdx.x + 2];
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 1];  
					s_LLRsign[threadIdx.x] ^= s_LLRsign[threadIdx.x + 1];
			}
			else{  //9 12 15
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 8];  
					s_LLRsign[threadIdx.x] ^= s_LLRsign[threadIdx.x + 8];
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 4];  
					s_LLRsign[threadIdx.x] ^= s_LLRsign[threadIdx.x + 4];
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 2];  
					s_LLRsign[threadIdx.x] ^= s_LLRsign[threadIdx.x + 2];
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 1];  
					s_LLRsign[threadIdx.x] ^= s_LLRsign[threadIdx.x + 1];
			}
			break;
		case 31:    //3x24  6x744  9x24  //CIF rowSum 3x96, 5x15, 6x2946, 7x15, 9x96
			if(rowDeg <= 4 ){  //3  
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 2];  
					s_LLRsign[threadIdx.x] ^= s_LLRsign[threadIdx.x + 2];
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 1];  
					s_LLRsign[threadIdx.x] ^= s_LLRsign[threadIdx.x + 1];
			}
			else if(rowDeg <= 8){  //6
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 4];  
					s_LLRsign[threadIdx.x] ^= s_LLRsign[threadIdx.x + 4];
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 2];  
					s_LLRsign[threadIdx.x] ^= s_LLRsign[threadIdx.x + 2];
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 1];  
					s_LLRsign[threadIdx.x] ^= s_LLRsign[threadIdx.x + 1];
			}
			else{  //9
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 8];  
					s_LLRsign[threadIdx.x] ^= s_LLRsign[threadIdx.x + 8];
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 4];  
					s_LLRsign[threadIdx.x] ^= s_LLRsign[threadIdx.x + 4];
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 2];  
					s_LLRsign[threadIdx.x] ^= s_LLRsign[threadIdx.x + 2];
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 1];  
					s_LLRsign[threadIdx.x] ^= s_LLRsign[threadIdx.x + 1];
			}
			break;
		case 32:    //3x48  6x768  //CIF rowSum 3x96, 5x15, 6x2946, 7x15, 9x96
			if(Size==CIF){
				if(rowDeg <= 4 ){  //3  
						s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 2];  
						s_LLRsign[threadIdx.x] ^= s_LLRsign[threadIdx.x + 2];
						s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 1];  
						s_LLRsign[threadIdx.x] ^= s_LLRsign[threadIdx.x + 1];
				}
				else if(rowDeg <= 8){  //6
						s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 4];  
						s_LLRsign[threadIdx.x] ^= s_LLRsign[threadIdx.x + 4];
						s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 2];  
						s_LLRsign[threadIdx.x] ^= s_LLRsign[threadIdx.x + 2];
						s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 1];  
						s_LLRsign[threadIdx.x] ^= s_LLRsign[threadIdx.x + 1];
				}
				else{  //9
						s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 8];  
						s_LLRsign[threadIdx.x] ^= s_LLRsign[threadIdx.x + 8];
						s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 4];  
						s_LLRsign[threadIdx.x] ^= s_LLRsign[threadIdx.x + 4];
						s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 2];  
						s_LLRsign[threadIdx.x] ^= s_LLRsign[threadIdx.x + 2];
						s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 1];  
						s_LLRsign[threadIdx.x] ^= s_LLRsign[threadIdx.x + 1];
				}
				break;
			}
		case 33:    //3x96  6x744  //CIF rowSum 3x192, 5x15, 6x3042, 7x15
		case 34:    //3x144  6x720	  //CIF rowSum 3x384, 5x15, 6x2946, 7x15
		case 35:    //3x192  6x696	  //CIF rowSum 3x576, 5x15, 6x2850, 7x15			
		case 36:    //3x240  6x672   //CIF rowSum 3x960, 5x15, 6x2658, 7x15
		case 37:    //3x288  6x648	//CIF rowSum 3x1152, 5x15, 6x2562, 7x15		
		case 38:    //3x336  6x624	//CIF rowSum 3x1344, 5x15, 6x2466, 7x15	
		case 39:    //3x384  6x600	//CIF rowSum 3x1536, 5x15, 6x2370, 7x15		
		case 40:    //3x432  6x576	//CIF rowSum 3x1728, 5x15, 6x2274, 7x15			
		case 41:    //3x480  6x552	//CIF rowSum 3x1920, 5x15, 6x2178, 7x15			
		case 42:    //3x528  6x528	//CIF rowSum 3x2112, 5x15, 6x2082, 7x15			
		case 43:    //3x576  6x504	//CIF rowSum 3x2304, 5x15, 6x1986, 7x15			
		case 44:    //3x624  6x480	//CIF rowSum 3x2496, 5x15, 6x1890, 7x15			
		case 45:    //3x672  6x456	//CIF rowSum 3x2688, 5x15, 6x1794, 7x15			
		case 46:    //3x720  6x432	//CIF rowSum 3x2880, 5x15, 6x1698, 7x15			
		case 47:    //3x768  6x408	//CIF rowSum 3x3072, 5x15, 6x1602, 7x15			
		case 48:    //3x816  6x384	//CIF rowSum 3x3264, 5x15, 6x1506, 7x15			
		case 49:    //3x864  6x360	//CIF rowSum 3x3456, 5x15, 6x1410, 7x15			
		case 50:    //3x912  6x336	//CIF rowSum 3x3648, 5x15, 6x1314, 7x15			
		case 51:    //3x960  6x312	//CIF rowSum 3x3840, 5x15, 6x1218, 7x15			
		case 52:    //3x1008  6x288	//CIF rowSum 3x4032, 5x15, 6x1122, 7x15			
		case 53:    //3x1056  6x264	//CIF rowSum 3x4224, 5x15, 6x1026, 7x15			
		case 54:    //3x1104  6x240	//CIF rowSum 3x4416, 5x15, 6x930, 7x15			
		case 55:    //3x1152  6x216	//CIF rowSum 3x4608, 5x15, 6x834, 7x15			
		case 56:    //3x1200  6x192	//CIF rowSum 3x4800, 5x15, 6x738, 7x15			
		case 57:    //3x1248  6x168	 //CIF rowSum 3x4992, 5x15, 6x642, 7x15			
		case 58:    //3x1296  6x144	//CIF rowSum 3x5184, 5x15, 6x546, 7x15			
			if(rowDeg <= 4 ){  //3  
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 2];  
					s_LLRsign[threadIdx.x] ^= s_LLRsign[threadIdx.x + 2];
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 1];  
					s_LLRsign[threadIdx.x] ^= s_LLRsign[threadIdx.x + 1];
			}
			else{  //6
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 4];  
					s_LLRsign[threadIdx.x] ^= s_LLRsign[threadIdx.x + 4];
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 2];  
					s_LLRsign[threadIdx.x] ^= s_LLRsign[threadIdx.x + 2];
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 1];  
					s_LLRsign[threadIdx.x] ^= s_LLRsign[threadIdx.x + 1];
			}
			break;
		case 59:    //3x1344  6x120	//CIF rowSum 2x5, 3x5364, 4x7, 5x10, 6x462, 7x8			
		case 60:    //3x1392  6x96	//CIF rowSum 2x15, 3x5538, 4x15, 6x384			
		case 61:    //3x1440  6x72	//CIF rowSum 2x15, 3x5730, 4x15, 6x288			
		case 62:    //3x1488  6x48	//CIF rowSum 2x15, 3x5922, 4x15, 6x192			
		case 63:    //3x1536  6x24  //CIF rowSum 2x15, 3x6114, 4x15, 6x96
			if(Size==QCIF){
				if(rowDeg <= 4 ){  //3  
						s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 2];  
						s_LLRsign[threadIdx.x] ^= s_LLRsign[threadIdx.x + 2];
						s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 1];  
						s_LLRsign[threadIdx.x] ^= s_LLRsign[threadIdx.x + 1];
				}
				else{  //6
						s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 4];  
						s_LLRsign[threadIdx.x] ^= s_LLRsign[threadIdx.x + 4];
						s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 2];  
						s_LLRsign[threadIdx.x] ^= s_LLRsign[threadIdx.x + 2];
						s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 1];  
						s_LLRsign[threadIdx.x] ^= s_LLRsign[threadIdx.x + 1];
				}
				break;
			}
			else{
				if(rowDeg == 2){
						s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 1];
						s_LLRsign[threadIdx.x] ^= s_LLRsign[threadIdx.x + 1];
				}
				else if(rowDeg <= 4 ){  //3  4
						s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 2];  
						s_LLRsign[threadIdx.x] ^= s_LLRsign[threadIdx.x + 2];
						s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 1];  
						s_LLRsign[threadIdx.x] ^= s_LLRsign[threadIdx.x + 1];
				}
				else{  //5 6 7
						s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 4];  
						s_LLRsign[threadIdx.x] ^= s_LLRsign[threadIdx.x + 4];
						s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 2];  
						s_LLRsign[threadIdx.x] ^= s_LLRsign[threadIdx.x + 2];
						s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x + 1];  
						s_LLRsign[threadIdx.x] ^= s_LLRsign[threadIdx.x + 1];
				}
				break;
			}			
		/*default:
			__syncthreads();
			do{
				if(rowPos < half){
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x+half];  
					s_LLRsign[threadIdx.x] ^= s_LLRsign[threadIdx.x+half];
				}
				half >>= 1;
				__syncthreads();
			}while(half);				
			break;*/
	}
}

template <int redType, int Size> __device__ void 
reduction(int rowPos, int rowDeg,volatile int s_check[LDPCA_BLOCK_SIZE]){
	/* Reduction : Compeletely Unroll for each Ladder size*/
	switch (redType){
		case 0:  //rowSum 99x48
			__syncthreads();
			if (rowPos <  64){
				s_check[threadIdx.x] ^= s_check[threadIdx.x + 64];					
			}
			__syncthreads();
			if (rowPos <  32){
				s_check[threadIdx.x] ^= s_check[threadIdx.x + 32];
				#if __CUDA_ARCH__ >= 200
					s_check[threadIdx.x] = __popc(__ballot(s_check[threadIdx.x])) & 1; // Magic happens here
				#else
					s_check[threadIdx.x] ^= s_check[threadIdx.x + 16];
					s_check[threadIdx.x] ^= s_check[threadIdx.x + 8];
					s_check[threadIdx.x] ^= s_check[threadIdx.x + 4];
					s_check[threadIdx.x] ^= s_check[threadIdx.x + 2];
					s_check[threadIdx.x] ^= s_check[threadIdx.x + 1];
				#endif
			}				

			__syncthreads();
			break;
		case 1:  //rowSum 48x24, 51x24, 99x24
			__syncthreads();
			if(rowDeg > 64 ){  //99
				if( rowPos < 64){
					s_check[threadIdx.x] ^= s_check[threadIdx.x + 64];	
				}
				__syncthreads();
			}
			if (rowPos < 32){
				s_check[threadIdx.x] ^= s_check[threadIdx.x + 32];
				#if __CUDA_ARCH__ >= 200
					s_check[threadIdx.x] = __popc(__ballot(s_check[threadIdx.x])) & 1; // Magic happens here
				#else
					s_check[threadIdx.x] ^= s_check[threadIdx.x + 16];
					s_check[threadIdx.x] ^= s_check[threadIdx.x + 8];
					s_check[threadIdx.x] ^= s_check[threadIdx.x + 4];
					s_check[threadIdx.x] ^= s_check[threadIdx.x + 2];
					s_check[threadIdx.x] ^= s_check[threadIdx.x + 1];
				#endif
			}
			__syncthreads();
			break;
		case 2:   //rowSum 48x48  51x48
			__syncthreads();
			if (rowPos < 32){
				s_check[threadIdx.x] ^= s_check[threadIdx.x + 32];
				#if __CUDA_ARCH__ >= 200
					s_check[threadIdx.x] = __popc(__ballot(s_check[threadIdx.x])) & 1; // Magic happens here
				#else
					s_check[threadIdx.x] ^= s_check[threadIdx.x + 16];
					s_check[threadIdx.x] ^= s_check[threadIdx.x + 8];
					s_check[threadIdx.x] ^= s_check[threadIdx.x + 4];
					s_check[threadIdx.x] ^= s_check[threadIdx.x + 2];
					s_check[threadIdx.x] ^= s_check[threadIdx.x + 1];
				#endif
			}
			__syncthreads();
			break;
		case 3:   //rowSum 24x24  27x24  48x48  51x24
		case 4:   //rowSum 24x48  27x48  48x48
		case 5:   //rowSum 24x96  27x48  48x24
			if(rowDeg <= 32 ){  //24  27
				#if __CUDA_ARCH__ >= 200
					s_check[threadIdx.x] = __popc(__ballot(s_check[threadIdx.x])) & 1; // Magic happens here
				#else
					s_check[threadIdx.x] ^= s_check[threadIdx.x + 16];
					s_check[threadIdx.x] ^= s_check[threadIdx.x + 8];
					s_check[threadIdx.x] ^= s_check[threadIdx.x + 4];
					s_check[threadIdx.x] ^= s_check[threadIdx.x + 2];
					s_check[threadIdx.x] ^= s_check[threadIdx.x + 1];
				#endif
			}
			else{  //48 51
				__syncthreads();
				if (rowPos < 32){
					s_check[threadIdx.x] ^= s_check[threadIdx.x + 32];
					#if __CUDA_ARCH__ >= 200
						s_check[threadIdx.x] = __popc(__ballot(s_check[threadIdx.x])) & 1; // Magic happens here
					#else
						s_check[threadIdx.x] ^= s_check[threadIdx.x + 16];
						s_check[threadIdx.x] ^= s_check[threadIdx.x + 8];
						s_check[threadIdx.x] ^= s_check[threadIdx.x + 4];
						s_check[threadIdx.x] ^= s_check[threadIdx.x + 2];
						s_check[threadIdx.x] ^= s_check[threadIdx.x + 1];
					#endif
				}
				__syncthreads();
			}
			break;
		case 6:   //rowSum 24x144  27x48
			#if __CUDA_ARCH__ >= 200
				s_check[threadIdx.x] = __popc(__ballot(s_check[threadIdx.x])) & 1; // Magic happens here
			#else
				s_check[threadIdx.x] ^= s_check[threadIdx.x + 16];
				s_check[threadIdx.x] ^= s_check[threadIdx.x + 8];
				s_check[threadIdx.x] ^= s_check[threadIdx.x + 4];
				s_check[threadIdx.x] ^= s_check[threadIdx.x + 2];
				s_check[threadIdx.x] ^= s_check[threadIdx.x + 1];
			#endif
			break;
		case 7:    //12x24  15x24  24x144  27x24
		case 8:    //12x48  15x48  24x144
		case 9:    //12x96  15x48  24x120
		case 10:   //12x144 15x48  24x96
		case 11:   //12x192 15x48  24x72
		case 12:   //12x240 15x48  24x48
		case 13:   //12x288 15x48  24x24
			if(rowDeg <= 16 ){  //12  15
					s_check[threadIdx.x] ^= s_check[threadIdx.x + 8];
					s_check[threadIdx.x] ^= s_check[threadIdx.x + 4];
					s_check[threadIdx.x] ^= s_check[threadIdx.x + 2];
					s_check[threadIdx.x] ^= s_check[threadIdx.x + 1];
			}
			else{  //24 27
				#if __CUDA_ARCH__ >= 200
					s_check[threadIdx.x] = __popc(__ballot(s_check[threadIdx.x])) & 1; // Magic happens here
				#else
					s_check[threadIdx.x] ^= s_check[threadIdx.x + 16];
					s_check[threadIdx.x] ^= s_check[threadIdx.x + 8];
					s_check[threadIdx.x] ^= s_check[threadIdx.x + 4];
					s_check[threadIdx.x] ^= s_check[threadIdx.x + 2];
					s_check[threadIdx.x] ^= s_check[threadIdx.x + 1];
				#endif
			}
			break;
		case 14:    //12x336  15x48
				s_check[threadIdx.x] ^= s_check[threadIdx.x + 8];
				s_check[threadIdx.x] ^= s_check[threadIdx.x + 4];
				s_check[threadIdx.x] ^= s_check[threadIdx.x + 2];
				s_check[threadIdx.x] ^= s_check[threadIdx.x + 1];
			break;
		case 15:    //6x24  9x24  12x336  15x24
		case 16:    //6x48  9x48  12x336
		case 17:    //6x96  9x48  12x312
		case 18:    //6x144  9x48  12x288
		case 19:    //6x192  9x48  12x264
		case 20:    //6x240  9x48  12x240
		case 21:    //6x288  9x48  12x216
		case 22:    //6x336  9x48  12x192
		case 23:    //6x384  9x48  12x168
		case 24:    //6x432  9x48  12x144
		case 25:    //6x480  9x48  12x120
		case 26:    //6x528  9x48  12x96
		case 27:    //6x576  9x48  12x72
		case 28:    //6x624  9x48  12x48
		case 29:    //6x672  9x48  12x24
		case 30:    //6x720  9x48
			if(rowDeg <= 8 ){  //6
					s_check[threadIdx.x] ^= s_check[threadIdx.x + 4];
					s_check[threadIdx.x] ^= s_check[threadIdx.x + 2];
					s_check[threadIdx.x] ^= s_check[threadIdx.x + 1];
			}
			else{  //9 12 15
					s_check[threadIdx.x] ^= s_check[threadIdx.x + 8];
					s_check[threadIdx.x] ^= s_check[threadIdx.x + 4];
					s_check[threadIdx.x] ^= s_check[threadIdx.x + 2];
					s_check[threadIdx.x] ^= s_check[threadIdx.x + 1];
			}
			break;
		case 31:    //3x24  6x744  9x24
			if(rowDeg <= 4 ){  //3  
					s_check[threadIdx.x] ^= s_check[threadIdx.x + 2];
					s_check[threadIdx.x] ^= s_check[threadIdx.x + 1];
			}
			else if(rowDeg <= 8){  //6
					s_check[threadIdx.x] ^= s_check[threadIdx.x + 4];
					s_check[threadIdx.x] ^= s_check[threadIdx.x + 2];
					s_check[threadIdx.x] ^= s_check[threadIdx.x + 1];
			}
			else{  //9
					s_check[threadIdx.x] ^= s_check[threadIdx.x + 8];
					s_check[threadIdx.x] ^= s_check[threadIdx.x + 4];
					s_check[threadIdx.x] ^= s_check[threadIdx.x + 2];
					s_check[threadIdx.x] ^= s_check[threadIdx.x + 1];
			}
			break;
		case 32:    //3x48  6x768  //CIF rowSum 3x96, 5x15, 6x2946, 7x15, 9x96
			if(Size==CIF){
				if(rowDeg <= 4 ){  //3  
						s_check[threadIdx.x] ^= s_check[threadIdx.x + 2];
						s_check[threadIdx.x] ^= s_check[threadIdx.x + 1];
				}
				else if(rowDeg <= 8){  //6
						s_check[threadIdx.x] ^= s_check[threadIdx.x + 4];
						s_check[threadIdx.x] ^= s_check[threadIdx.x + 2];
						s_check[threadIdx.x] ^= s_check[threadIdx.x + 1];
				}
				else{  //9
						s_check[threadIdx.x] ^= s_check[threadIdx.x + 8];
						s_check[threadIdx.x] ^= s_check[threadIdx.x + 4];
						s_check[threadIdx.x] ^= s_check[threadIdx.x + 2];
						s_check[threadIdx.x] ^= s_check[threadIdx.x + 1];
				}
				break;
			}
		case 33:    //3x96  6x744
		case 34:    //3x144  6x720		
		case 35:    //3x192  6x696				
		case 36:    //3x240  6x672				
		case 37:    //3x288  6x648				
		case 38:    //3x336  6x624				
		case 39:    //3x384  6x600				
		case 40:    //3x432  6x576				
		case 41:    //3x480  6x552				
		case 42:    //3x528  6x528				
		case 43:    //3x576  6x504				
		case 44:    //3x624  6x480				
		case 45:    //3x672  6x456				
		case 46:    //3x720  6x432				
		case 47:    //3x768  6x408				
		case 48:    //3x816  6x384				
		case 49:    //3x864  6x360				
		case 50:    //3x912  6x336				
		case 51:    //3x960  6x312				
		case 52:    //3x1008  6x288				
		case 53:    //3x1056  6x264				
		case 54:    //3x1104  6x240				
		case 55:    //3x1152  6x216				
		case 56:    //3x1200  6x192				
		case 57:    //3x1248  6x168				
		case 58:    //3x1296  6x144				
			if(rowDeg <= 4 ){  //3  
					s_check[threadIdx.x] ^= s_check[threadIdx.x + 2];
					s_check[threadIdx.x] ^= s_check[threadIdx.x + 1];
			}
			else{  //6
					s_check[threadIdx.x] ^= s_check[threadIdx.x + 4];
					s_check[threadIdx.x] ^= s_check[threadIdx.x + 2];
					s_check[threadIdx.x] ^= s_check[threadIdx.x + 1];
			}
			break;
		case 59:    //3x1344  6x120	//CIF rowSum 2x5, 3x5364, 4x7, 5x10, 6x462, 7x8			
		case 60:    //3x1392  6x96	//CIF rowSum 2x15, 3x5538, 4x15, 6x384			
		case 61:    //3x1440  6x72	//CIF rowSum 2x15, 3x5730, 4x15, 6x288			
		case 62:    //3x1488  6x48	//CIF rowSum 2x15, 3x5922, 4x15, 6x192			
		case 63:    //3x1536  6x24  //CIF rowSum 2x15, 3x6114, 4x15, 6x96
			if(Size==QCIF){
				if(rowDeg <= 4 ){  //3  
						s_check[threadIdx.x] ^= s_check[threadIdx.x + 2];
						s_check[threadIdx.x] ^= s_check[threadIdx.x + 1];
				}
				else{  //6
						s_check[threadIdx.x] ^= s_check[threadIdx.x + 4];
						s_check[threadIdx.x] ^= s_check[threadIdx.x + 2];
						s_check[threadIdx.x] ^= s_check[threadIdx.x + 1];
				}
				break;
			}
			else{
				if(rowDeg == 2){
						s_check[threadIdx.x] ^= s_check[threadIdx.x + 1];
				}
				else if(rowDeg <= 4 ){  //3  4
						s_check[threadIdx.x] ^= s_check[threadIdx.x + 2];
						s_check[threadIdx.x] ^= s_check[threadIdx.x + 1];
				}
				else{  //5 6 7
						s_check[threadIdx.x] ^= s_check[threadIdx.x + 4];
						s_check[threadIdx.x] ^= s_check[threadIdx.x + 2];
						s_check[threadIdx.x] ^= s_check[threadIdx.x + 1];
				}
				break;
			}			
		/*default:
			__syncthreads();
			do{
				if(rowPos < half){
					s_LLRmag[threadIdx.x] += s_LLRmag[threadIdx.x+half];  
					s_check[threadIdx.x] ^= s_check[threadIdx.x+half];
				}
				half >>= 1;
				__syncthreads();
			}while(half);				
			break;*/
	}
}


#ifdef LDPCA_SINGLE_BUFFER
void dummy(){ //let compiler initialize each function instance (CUDA drvier API)
	InitMessage_kernal<0, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<1, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<2, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<3, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<4, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<5, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<6, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<7, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<8, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<9, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<10, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<11, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<12, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<13, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<14, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<15, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<16, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<17, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<18, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<19, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<20, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<21, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<22, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<23, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<24, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<25, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<26, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<27, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<28, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<29, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<30, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<31, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<32, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<33, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<34, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<35, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<36, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<37, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<38, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<39, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<40, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<41, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<42, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<43, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<44, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<45, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<46, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<47, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<48, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<49, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<50, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<51, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<52, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<53, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<54, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<55, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<56, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<57, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<58, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<59, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<60, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<61, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<62, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<63, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	UpdateMessage_kernal<0, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<1, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<2, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<3, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<4, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<5, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<6, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<7, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<8, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<9, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<10, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<11, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<12, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<13, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<14, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<15, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<16, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<17, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<18, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<19, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<20, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<21, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<22, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<23, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<24, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<25, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<26, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<27, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<28, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<29, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<30, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<31, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<32, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<33, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<34, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<35, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<36, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<37, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<38, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<39, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<40, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<41, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<42, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<43, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<44, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<45, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<46, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<47, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<48, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<49, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<50, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<51, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<52, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<53, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<54, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<55, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<56, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<57, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<58, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<59, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<60, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<61, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<62, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<63, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	CheckUpdateMessage_kernal<0, QCIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<1, QCIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<2, QCIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<3, QCIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<4, QCIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<5, QCIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<6, QCIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<7, QCIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<8, QCIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<9, QCIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<10, QCIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<11, QCIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<12, QCIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<13, QCIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<14, QCIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<15, QCIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<16, QCIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<17, QCIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<18, QCIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<19, QCIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<20, QCIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<21, QCIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<22, QCIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<23, QCIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<24, QCIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<25, QCIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<26, QCIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<27, QCIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<28, QCIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<29, QCIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<30, QCIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<31, QCIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<32, QCIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<33, QCIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<34, QCIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<35, QCIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<36, QCIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<37, QCIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<38, QCIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<39, QCIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<40, QCIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<41, QCIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<42, QCIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<43, QCIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<44, QCIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<45, QCIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<46, QCIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<47, QCIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<48, QCIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<49, QCIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<50, QCIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<51, QCIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<52, QCIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<53, QCIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<54, QCIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<55, QCIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<56, QCIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<57, QCIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<58, QCIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<59, QCIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<60, QCIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<61, QCIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<62, QCIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<63, QCIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckMessage_kernal<0, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<1, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<2, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<3, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<4, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<5, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<6, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<7, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<8, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<9, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<10, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<11, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<12, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<13, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<14, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<15, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<16, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<17, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<18, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<19, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<20, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<21, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<22, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<23, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<24, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<25, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<26, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<27, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<28, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<29, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<30, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<31, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<32, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<33, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<34, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<35, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<36, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<37, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<38, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<39, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<40, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<41, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<42, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<43, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<44, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<45, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<46, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<47, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<48, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<49, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<50, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<51, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<52, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<53, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<54, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<55, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<56, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<57, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<58, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<59, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<60, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<61, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<62, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<63, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	InitMessage_kernal<0, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<1, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<2, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<3, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<4, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<5, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<6, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<7, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<8, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<9, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<10, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<11, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<12, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<13, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<14, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<15, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<16, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<17, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<18, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<19, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<20, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<21, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<22, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<23, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<24, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<25, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<26, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<27, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<28, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<29, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<30, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<31, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<32, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<33, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<34, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<35, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<36, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<37, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<38, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<39, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<40, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<41, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<42, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<43, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<44, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<45, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<46, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<47, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<48, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<49, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<50, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<51, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<52, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<53, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<54, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<55, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<56, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<57, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<58, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<59, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<60, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<61, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<62, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<63, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	UpdateMessage_kernal<0, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<1, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<2, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<3, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<4, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<5, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<6, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<7, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<8, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<9, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<10, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<11, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<12, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<13, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<14, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<15, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<16, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<17, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<18, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<19, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<20, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<21, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<22, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<23, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<24, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<25, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<26, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<27, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<28, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<29, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<30, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<31, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<32, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<33, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<34, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<35, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<36, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<37, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<38, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<39, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<40, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<41, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<42, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<43, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<44, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<45, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<46, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<47, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<48, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<49, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<50, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<51, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<52, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<53, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<54, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<55, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<56, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<57, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<58, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<59, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<60, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<61, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<62, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<63, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL );
	CheckUpdateMessage_kernal<0, CIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<1, CIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<2, CIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<3, CIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<4, CIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<5, CIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<6, CIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<7, CIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<8, CIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<9, CIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<10, CIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<11, CIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<12, CIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<13, CIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<14, CIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<15, CIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<16, CIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<17, CIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<18, CIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<19, CIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<20, CIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<21, CIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<22, CIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<23, CIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<24, CIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<25, CIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<26, CIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<27, CIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<28, CIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<29, CIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<30, CIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<31, CIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<32, CIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<33, CIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<34, CIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<35, CIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<36, CIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<37, CIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<38, CIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<39, CIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<40, CIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<41, CIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<42, CIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<43, CIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<44, CIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<45, CIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<46, CIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<47, CIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<48, CIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<49, CIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<50, CIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<51, CIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<52, CIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<53, CIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<54, CIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<55, CIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<56, CIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<57, CIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<58, CIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<59, CIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<60, CIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<61, CIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<62, CIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<63, CIF> <<< 1, 1 >>>( NULL, NULL, NULL, NULL, NULL, NULL);
	CheckMessage_kernal<0, CIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<1, CIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<2, CIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<3, CIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<4, CIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<5, CIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<6, CIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<7, CIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<8, CIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<9, CIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<10, CIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<11, CIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<12, CIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<13, CIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<14, CIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<15, CIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<16, CIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<17, CIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<18, CIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<19, CIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<20, CIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<21, CIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<22, CIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<23, CIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<24, CIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<25, CIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<26, CIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<27, CIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<28, CIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<29, CIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<30, CIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<31, CIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<32, CIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<33, CIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<34, CIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<35, CIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<36, CIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<37, CIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<38, CIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<39, CIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<40, CIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<41, CIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<42, CIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<43, CIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<44, CIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<45, CIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<46, CIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<47, CIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<48, CIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<49, CIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<50, CIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<51, CIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<52, CIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<53, CIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<54, CIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<55, CIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<56, CIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<57, CIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<58, CIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<59, CIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<60, CIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<61, CIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<62, CIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
	CheckMessage_kernal<63, CIF> <<< 1, 1 >>> (NULL, NULL, NULL,   NULL,NULL);
}
#else
void dummy(){ //let compiler initialize each function instance (CUDA drvier API)
	InitMessage_kernal<0, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<1, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<2, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<3, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<4, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<5, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<6, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<7, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<8, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<9, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<10, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<11, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<12, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<13, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<14, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<15, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<16, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<17, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<18, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<19, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<20, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<21, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<22, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<23, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<24, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<25, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<26, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<27, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<28, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<29, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<30, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<31, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<32, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<33, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<34, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<35, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<36, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<37, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<38, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<39, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<40, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<41, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<42, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<43, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<44, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<45, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<46, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<47, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<48, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<49, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<50, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<51, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<52, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<53, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<54, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<55, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<56, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<57, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<58, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<59, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<60, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<61, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<62, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<63, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	UpdateMessage_kernal<0, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<1, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<2, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<3, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<4, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<5, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<6, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<7, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<8, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<9, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<10, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<11, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<12, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<13, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<14, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<15, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<16, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<17, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<18, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<19, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<20, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<21, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<22, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<23, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<24, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<25, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<26, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<27, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<28, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<29, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<30, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<31, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<32, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<33, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<34, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<35, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<36, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<37, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<38, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<39, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<40, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<41, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<42, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<43, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<44, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<45, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<46, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<47, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<48, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<49, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<50, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<51, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<52, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<53, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<54, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<55, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<56, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<57, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<58, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<59, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<60, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<61, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<62, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<63, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	CheckUpdateMessage_kernal<0, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<1, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<2, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<3, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<4, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<5, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<6, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<7, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<8, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<9, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<10, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<11, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<12, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<13, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<14, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<15, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<16, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<17, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<18, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<19, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<20, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<21, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<22, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<23, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<24, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<25, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<26, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<27, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<28, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<29, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<30, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<31, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<32, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<33, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<34, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<35, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<36, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<37, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<38, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<39, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<40, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<41, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<42, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<43, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<44, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<45, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<46, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<47, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<48, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<49, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<50, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<51, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<52, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<53, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<54, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<55, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<56, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<57, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<58, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<59, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<60, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<61, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<62, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<63, QCIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckMessage_kernal<0, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<1, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<2, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<3, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<4, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<5, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<6, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<7, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<8, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<9, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<10, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<11, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<12, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<13, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<14, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<15, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<16, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<17, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<18, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<19, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<20, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<21, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<22, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<23, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<24, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<25, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<26, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<27, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<28, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<29, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<30, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<31, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<32, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<33, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<34, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<35, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<36, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<37, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<38, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<39, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<40, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<41, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<42, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<43, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<44, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<45, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<46, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<47, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<48, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<49, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<50, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<51, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<52, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<53, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<54, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<55, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<56, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<57, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<58, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<59, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<60, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<61, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<62, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<63, QCIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	InitMessage_kernal<0, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<1, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<2, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<3, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<4, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<5, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<6, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<7, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<8, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<9, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<10, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<11, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<12, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<13, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<14, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<15, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<16, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<17, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<18, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<19, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<20, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<21, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<22, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<23, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<24, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<25, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<26, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<27, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<28, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<29, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<30, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<31, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<32, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<33, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<34, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<35, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<36, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<37, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<38, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<39, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<40, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<41, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<42, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<43, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<44, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<45, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<46, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<47, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<48, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<49, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<50, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<51, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<52, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<53, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<54, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<55, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<56, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<57, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<58, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<59, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<60, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<61, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<62, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	InitMessage_kernal<63, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL);
	UpdateMessage_kernal<0, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<1, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<2, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<3, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<4, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<5, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<6, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<7, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<8, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<9, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<10, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<11, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<12, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<13, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<14, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<15, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<16, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<17, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<18, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<19, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<20, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<21, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<22, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<23, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<24, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<25, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<26, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<27, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<28, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<29, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<30, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<31, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<32, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<33, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<34, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<35, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<36, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<37, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<38, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<39, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<40, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<41, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<42, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<43, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<44, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<45, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<46, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<47, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<48, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<49, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<50, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<51, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<52, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<53, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<54, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<55, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<56, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<57, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<58, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<59, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<60, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<61, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<62, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	UpdateMessage_kernal<63, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL );
	CheckUpdateMessage_kernal<0, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<1, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<2, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<3, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<4, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<5, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<6, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<7, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<8, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<9, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<10, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<11, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<12, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<13, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<14, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<15, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<16, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<17, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<18, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<19, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<20, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<21, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<22, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<23, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<24, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<25, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<26, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<27, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<28, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<29, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<30, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<31, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<32, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<33, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<34, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<35, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<36, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<37, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<38, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<39, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<40, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<41, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<42, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<43, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<44, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<45, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<46, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<47, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<48, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<49, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<50, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<51, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<52, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<53, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<54, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<55, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<56, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<57, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<58, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<59, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<60, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<61, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<62, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckUpdateMessage_kernal<63, CIF> <<< 1, 1 >>>(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	CheckMessage_kernal<0, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<1, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<2, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<3, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<4, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<5, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<6, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<7, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<8, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<9, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<10, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<11, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<12, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<13, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<14, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<15, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<16, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<17, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<18, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<19, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<20, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<21, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<22, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<23, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<24, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<25, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<26, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<27, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<28, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<29, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<30, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<31, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<32, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<33, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<34, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<35, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<36, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<37, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<38, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<39, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<40, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<41, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<42, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<43, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<44, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<45, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<46, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<47, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<48, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<49, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<50, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<51, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<52, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<53, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<54, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<55, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<56, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<57, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<58, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<59, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<60, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<61, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<62, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
	CheckMessage_kernal<63, CIF> <<< 1, 1 >>> (NULL, NULL, NULL, NULL, NULL,NULL);
}

#endif

void bind_txture_RT(Msg* message, int size){
	cudaBindTexture( 0, tex_message, message, size );
}
void init_kernel_RT(CUstream stream, int nCode, int Size, int numBlock, Index2*   HinitIdx, char*  syndromeByte, float*  softInput, RowInfo*  Hinfo,  float*  softInputPadding, Msg*  InputMessage){	
	switch (nCode) {
		case 0:
			if(Size==QCIF)
				InitMessage_kernal<0, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			else
				InitMessage_kernal<0, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			break;
		case 1:
			if(Size==QCIF)
				InitMessage_kernal<1, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			else
				InitMessage_kernal<1, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			break;
		case 2:
			if(Size==QCIF)
				InitMessage_kernal<2, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			else
				InitMessage_kernal<2, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			break;
		case 3:
			if(Size==QCIF)
				InitMessage_kernal<3, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			else
				InitMessage_kernal<3, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			break;
		case 4:
			if(Size==QCIF)
				InitMessage_kernal<4, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			else
				InitMessage_kernal<4, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			break;
		case 5:
			if(Size==QCIF)
				InitMessage_kernal<5, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			else
				InitMessage_kernal<5, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			break;
		case 6:
			if(Size==QCIF)
				InitMessage_kernal<6, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			else
				InitMessage_kernal<6, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			break;
		case 7:
			if(Size==QCIF)
				InitMessage_kernal<7, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			else
				InitMessage_kernal<7, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			break;
		case 8:
			if(Size==QCIF)
				InitMessage_kernal<8, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			else
				InitMessage_kernal<8, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			break;
		case 9:
			if(Size==QCIF)
				InitMessage_kernal<9, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			else
				InitMessage_kernal<9, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			break;
		case 10:
			if(Size==QCIF)
				InitMessage_kernal<10, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			else
				InitMessage_kernal<10, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			break;
		case 11:
			if(Size==QCIF)
				InitMessage_kernal<11, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			else
				InitMessage_kernal<11, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			break;
		case 12:
			if(Size==QCIF)
				InitMessage_kernal<12, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			else
				InitMessage_kernal<12, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			break;
		case 13:
			if(Size==QCIF)
				InitMessage_kernal<13, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			else
				InitMessage_kernal<13, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			break;
		case 14:
			if(Size==QCIF)
				InitMessage_kernal<14, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			else
				InitMessage_kernal<14, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			break;
		case 15:
			if(Size==QCIF)
				InitMessage_kernal<15, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			else
				InitMessage_kernal<15, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			break;
		case 16:
			if(Size==QCIF)
				InitMessage_kernal<16, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			else
				InitMessage_kernal<16, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			break;
		case 17:
			if(Size==QCIF)
				InitMessage_kernal<17, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			else
				InitMessage_kernal<17, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			break;
		case 18:
			if(Size==QCIF)
				InitMessage_kernal<18, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			else
				InitMessage_kernal<18, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			break;
		case 19:
			if(Size==QCIF)
				InitMessage_kernal<19, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			else
				InitMessage_kernal<19, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			break;
		case 20:
			if(Size==QCIF)
				InitMessage_kernal<20, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			else
				InitMessage_kernal<20, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			break;
		case 21:
			if(Size==QCIF)
				InitMessage_kernal<21, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			else
				InitMessage_kernal<21, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			break;
		case 22:
			if(Size==QCIF)
				InitMessage_kernal<22, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			else
				InitMessage_kernal<22, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			break;
		case 23:
			if(Size==QCIF)
				InitMessage_kernal<23, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			else
				InitMessage_kernal<23, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			break;
		case 24:
			if(Size==QCIF)
				InitMessage_kernal<24, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			else
				InitMessage_kernal<24, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			break;
		case 25:
			if(Size==QCIF)
				InitMessage_kernal<25, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			else
				InitMessage_kernal<25, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			break;
		case 26:
			if(Size==QCIF)
				InitMessage_kernal<26, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			else
				InitMessage_kernal<26, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			break;
		case 27:
			if(Size==QCIF)
				InitMessage_kernal<27, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			else
				InitMessage_kernal<27, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			break;
		case 28:
			if(Size==QCIF)
				InitMessage_kernal<28, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			else
				InitMessage_kernal<28, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			break;
		case 29:
			if(Size==QCIF)
				InitMessage_kernal<29, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			else
				InitMessage_kernal<29, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			break;
		case 30:
			if(Size==QCIF)
				InitMessage_kernal<30, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			else
				InitMessage_kernal<30, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			break;
		case 31:
			if(Size==QCIF)
				InitMessage_kernal<31, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			else
				InitMessage_kernal<31, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			break;
		case 32:
			if(Size==QCIF)
				InitMessage_kernal<32, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			else
				InitMessage_kernal<32, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			break;
		case 33:
			if(Size==QCIF)
				InitMessage_kernal<33, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			else
				InitMessage_kernal<33, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			break;
		case 34:
			if(Size==QCIF)
				InitMessage_kernal<34, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			else
				InitMessage_kernal<34, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			break;
		case 35:
			if(Size==QCIF)
				InitMessage_kernal<35, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			else
				InitMessage_kernal<35, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			break;
		case 36:
			if(Size==QCIF)
				InitMessage_kernal<36, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			else
				InitMessage_kernal<36, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			break;
		case 37:
			if(Size==QCIF)
				InitMessage_kernal<37, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			else
				InitMessage_kernal<37, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			break;
		case 38:
			if(Size==QCIF)
				InitMessage_kernal<38, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			else
				InitMessage_kernal<38, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			break;
		case 39:
			if(Size==QCIF)
				InitMessage_kernal<39, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			else
				InitMessage_kernal<39, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			break;
		case 40:
			if(Size==QCIF)
				InitMessage_kernal<40, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			else
				InitMessage_kernal<40, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			break;
		case 41:
			if(Size==QCIF)
				InitMessage_kernal<41, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			else
				InitMessage_kernal<41, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			break;
		case 42:
			if(Size==QCIF)
				InitMessage_kernal<42, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			else
				InitMessage_kernal<42, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			break;
		case 43:
			if(Size==QCIF)
				InitMessage_kernal<43, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			else
				InitMessage_kernal<43, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			break;
		case 44:
			if(Size==QCIF)
				InitMessage_kernal<44, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			else
				InitMessage_kernal<44, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			break;
		case 45:
			if(Size==QCIF)
				InitMessage_kernal<45, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			else
				InitMessage_kernal<45, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			break;
		case 46:
			if(Size==QCIF)
				InitMessage_kernal<46, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			else
				InitMessage_kernal<46, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			break;
		case 47:
			if(Size==QCIF)
				InitMessage_kernal<47, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			else
				InitMessage_kernal<47, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			break;
		case 48:
			if(Size==QCIF)
				InitMessage_kernal<48, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			else
				InitMessage_kernal<48, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			break;
		case 49:
			if(Size==QCIF)
				InitMessage_kernal<49, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			else
				InitMessage_kernal<49, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			break;
		case 50:
			if(Size==QCIF)
				InitMessage_kernal<50, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			else
				InitMessage_kernal<50, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			break;
		case 51:
			if(Size==QCIF)
				InitMessage_kernal<51, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			else
				InitMessage_kernal<51, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			break;
		case 52:
			if(Size==QCIF)
				InitMessage_kernal<52, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			else
				InitMessage_kernal<52, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			break;
		case 53:
			if(Size==QCIF)
				InitMessage_kernal<53, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			else
				InitMessage_kernal<53, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			break;
		case 54:
			if(Size==QCIF)
				InitMessage_kernal<54, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			else
				InitMessage_kernal<54, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			break;
		case 55:
			if(Size==QCIF)
				InitMessage_kernal<55, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			else
				InitMessage_kernal<55, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			break;
		case 56:
			if(Size==QCIF)
				InitMessage_kernal<56, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			else
				InitMessage_kernal<56, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			break;
		case 57:
			if(Size==QCIF)
				InitMessage_kernal<57, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			else
				InitMessage_kernal<57, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			break;
		case 58:
			if(Size==QCIF)
				InitMessage_kernal<58, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			else
				InitMessage_kernal<58, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			break;
		case 59:
			if(Size==QCIF)
				InitMessage_kernal<59, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			else
				InitMessage_kernal<59, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			break;
		case 60:
			if(Size==QCIF)
				InitMessage_kernal<60, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			else
				InitMessage_kernal<60, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			break;
		case 61:
			if(Size==QCIF)
				InitMessage_kernal<61, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			else
				InitMessage_kernal<61, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			break;
		case 62:
			if(Size==QCIF)
				InitMessage_kernal<62, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			else
				InitMessage_kernal<62, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			break;
		case 63:
			if(Size==QCIF)
				InitMessage_kernal<63, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			else
				InitMessage_kernal<63, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(HinitIdx, syndromeByte, softInput, Hinfo, softInputPadding, InputMessage);
			break;
	}
}
void update_kernel_RT(CUstream stream, int nCode, int Size, int numBlock, Msg*  InputMessage, float*  softInputPadding, ColInfo*  Vinfo, RowInfo*  Hinfo){
	switch (nCode) {
		case 0:
			if(Size==QCIF)
				UpdateMessage_kernal<0, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			else
				UpdateMessage_kernal<0, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			break;
		case 1:
			if(Size==QCIF)
				UpdateMessage_kernal<1, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			else
				UpdateMessage_kernal<1, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			break;
		case 2:
			if(Size==QCIF)
				UpdateMessage_kernal<2, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			else
				UpdateMessage_kernal<2, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			break;
		case 3:
			if(Size==QCIF)
				UpdateMessage_kernal<3, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			else
				UpdateMessage_kernal<3, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			break;
		case 4:
			if(Size==QCIF)
				UpdateMessage_kernal<4, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			else
				UpdateMessage_kernal<4, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			break;
		case 5:
			if(Size==QCIF)
				UpdateMessage_kernal<5, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			else
				UpdateMessage_kernal<5, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			break;
		case 6:
			if(Size==QCIF)
				UpdateMessage_kernal<6, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			else
				UpdateMessage_kernal<6, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			break;
		case 7:
			if(Size==QCIF)
				UpdateMessage_kernal<7, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			else
				UpdateMessage_kernal<7, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			break;
		case 8:
			if(Size==QCIF)
				UpdateMessage_kernal<8, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			else
				UpdateMessage_kernal<8, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			break;
		case 9:
			if(Size==QCIF)
				UpdateMessage_kernal<9, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			else
				UpdateMessage_kernal<9, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			break;
		case 10:
			if(Size==QCIF)
				UpdateMessage_kernal<10, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			else
				UpdateMessage_kernal<10, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			break;
		case 11:
			if(Size==QCIF)
				UpdateMessage_kernal<11, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			else
				UpdateMessage_kernal<11, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			break;
		case 12:
			if(Size==QCIF)
				UpdateMessage_kernal<12, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			else
				UpdateMessage_kernal<12, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			break;
		case 13:
			if(Size==QCIF)
				UpdateMessage_kernal<13, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			else
				UpdateMessage_kernal<13, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			break;
		case 14:
			if(Size==QCIF)
				UpdateMessage_kernal<14, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			else
				UpdateMessage_kernal<14, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			break;
		case 15:
			if(Size==QCIF)
				UpdateMessage_kernal<15, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			else
				UpdateMessage_kernal<15, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			break;
		case 16:
			if(Size==QCIF)
				UpdateMessage_kernal<16, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			else
				UpdateMessage_kernal<16, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			break;
		case 17:
			if(Size==QCIF)
				UpdateMessage_kernal<17, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			else
				UpdateMessage_kernal<17, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			break;
		case 18:
			if(Size==QCIF)
				UpdateMessage_kernal<18, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			else
				UpdateMessage_kernal<18, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			break;
		case 19:
			if(Size==QCIF)
				UpdateMessage_kernal<19, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			else
				UpdateMessage_kernal<19, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			break;
		case 20:
			if(Size==QCIF)
				UpdateMessage_kernal<20, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			else
				UpdateMessage_kernal<20, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			break;
		case 21:
			if(Size==QCIF)
				UpdateMessage_kernal<21, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			else
				UpdateMessage_kernal<21, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			break;
		case 22:
			if(Size==QCIF)
				UpdateMessage_kernal<22, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			else
				UpdateMessage_kernal<22, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			break;
		case 23:
			if(Size==QCIF)
				UpdateMessage_kernal<23, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			else
				UpdateMessage_kernal<23, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			break;
		case 24:
			if(Size==QCIF)
				UpdateMessage_kernal<24, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			else
				UpdateMessage_kernal<24, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			break;
		case 25:
			if(Size==QCIF)
				UpdateMessage_kernal<25, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			else
				UpdateMessage_kernal<25, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			break;
		case 26:
			if(Size==QCIF)
				UpdateMessage_kernal<26, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			else
				UpdateMessage_kernal<26, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			break;
		case 27:
			if(Size==QCIF)
				UpdateMessage_kernal<27, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			else
				UpdateMessage_kernal<27, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			break;
		case 28:
			if(Size==QCIF)
				UpdateMessage_kernal<28, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			else
				UpdateMessage_kernal<28, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			break;
		case 29:
			if(Size==QCIF)
				UpdateMessage_kernal<29, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			else
				UpdateMessage_kernal<29, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			break;
		case 30:
			if(Size==QCIF)
				UpdateMessage_kernal<30, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			else
				UpdateMessage_kernal<30, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			break;
		case 31:
			if(Size==QCIF)
				UpdateMessage_kernal<31, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			else
				UpdateMessage_kernal<31, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			break;
		case 32:
			if(Size==QCIF)
				UpdateMessage_kernal<32, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			else
				UpdateMessage_kernal<32, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			break;
		case 33:
			if(Size==QCIF)
				UpdateMessage_kernal<33, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			else
				UpdateMessage_kernal<33, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			break;
		case 34:
			if(Size==QCIF)
				UpdateMessage_kernal<34, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			else
				UpdateMessage_kernal<34, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			break;
		case 35:
			if(Size==QCIF)
				UpdateMessage_kernal<35, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			else
				UpdateMessage_kernal<35, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			break;
		case 36:
			if(Size==QCIF)
				UpdateMessage_kernal<36, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			else
				UpdateMessage_kernal<36, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			break;
		case 37:
			if(Size==QCIF)
				UpdateMessage_kernal<37, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			else
				UpdateMessage_kernal<37, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			break;
		case 38:
			if(Size==QCIF)
				UpdateMessage_kernal<38, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			else
				UpdateMessage_kernal<38, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			break;
		case 39:
			if(Size==QCIF)
				UpdateMessage_kernal<39, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			else
				UpdateMessage_kernal<39, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			break;
		case 40:
			if(Size==QCIF)
				UpdateMessage_kernal<40, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			else
				UpdateMessage_kernal<40, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			break;
		case 41:
			if(Size==QCIF)
				UpdateMessage_kernal<41, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			else
				UpdateMessage_kernal<41, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			break;
		case 42:
			if(Size==QCIF)
				UpdateMessage_kernal<42, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			else
				UpdateMessage_kernal<42, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			break;
		case 43:
			if(Size==QCIF)
				UpdateMessage_kernal<43, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			else
				UpdateMessage_kernal<43, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			break;
		case 44:
			if(Size==QCIF)
				UpdateMessage_kernal<44, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			else
				UpdateMessage_kernal<44, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			break;
		case 45:
			if(Size==QCIF)
				UpdateMessage_kernal<45, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			else
				UpdateMessage_kernal<45, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			break;
		case 46:
			if(Size==QCIF)
				UpdateMessage_kernal<46, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			else
				UpdateMessage_kernal<46, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			break;
		case 47:
			if(Size==QCIF)
				UpdateMessage_kernal<47, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			else
				UpdateMessage_kernal<47, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			break;
		case 48:
			if(Size==QCIF)
				UpdateMessage_kernal<48, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			else
				UpdateMessage_kernal<48, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			break;
		case 49:
			if(Size==QCIF)
				UpdateMessage_kernal<49, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			else
				UpdateMessage_kernal<49, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			break;
		case 50:
			if(Size==QCIF)
				UpdateMessage_kernal<50, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			else
				UpdateMessage_kernal<50, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			break;
		case 51:
			if(Size==QCIF)
				UpdateMessage_kernal<51, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			else
				UpdateMessage_kernal<51, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			break;
		case 52:
			if(Size==QCIF)
				UpdateMessage_kernal<52, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			else
				UpdateMessage_kernal<52, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			break;
		case 53:
			if(Size==QCIF)
				UpdateMessage_kernal<53, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			else
				UpdateMessage_kernal<53, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			break;
		case 54:
			if(Size==QCIF)
				UpdateMessage_kernal<54, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			else
				UpdateMessage_kernal<54, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			break;
		case 55:
			if(Size==QCIF)
				UpdateMessage_kernal<55, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			else
				UpdateMessage_kernal<55, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			break;
		case 56:
			if(Size==QCIF)
				UpdateMessage_kernal<56, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			else
				UpdateMessage_kernal<56, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			break;
		case 57:
			if(Size==QCIF)
				UpdateMessage_kernal<57, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			else
				UpdateMessage_kernal<57, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			break;
		case 58:
			if(Size==QCIF)
				UpdateMessage_kernal<58, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			else
				UpdateMessage_kernal<58, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			break;
		case 59:
			if(Size==QCIF)
				UpdateMessage_kernal<59, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			else
				UpdateMessage_kernal<59, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			break;
		case 60:
			if(Size==QCIF)
				UpdateMessage_kernal<60, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			else
				UpdateMessage_kernal<60, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			break;
		case 61:
			if(Size==QCIF)
				UpdateMessage_kernal<61, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			else
				UpdateMessage_kernal<61, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			break;
		case 62:
			if(Size==QCIF)
				UpdateMessage_kernal<62, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			else
				UpdateMessage_kernal<62, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			break;
		case 63:
			if(Size==QCIF)
				UpdateMessage_kernal<63, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			else
				UpdateMessage_kernal<63, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(InputMessage, softInputPadding, Vinfo, Hinfo);
			break;
	}
}
void check_kernel_RT(CUstream stream, int nCode, int Size, int numBlock, int decodedHeaderIdx, const float*  softInputPadding, ColInfo*  Vinfo, RowInfo*  Hinfo, unsigned char*  decodedInfo){
	switch (nCode) {
		case 0:
			if(Size==QCIF)
				CheckMessage_kernal<0, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckMessage_kernal<0, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 1:
			if(Size==QCIF)
				CheckMessage_kernal<1, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckMessage_kernal<1, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 2:
			if(Size==QCIF)
				CheckMessage_kernal<2, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckMessage_kernal<2, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 3:
			if(Size==QCIF)
				CheckMessage_kernal<3, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckMessage_kernal<3, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 4:
			if(Size==QCIF)
				CheckMessage_kernal<4, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckMessage_kernal<4, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 5:
			if(Size==QCIF)
				CheckMessage_kernal<5, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckMessage_kernal<5, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 6:
			if(Size==QCIF)
				CheckMessage_kernal<6, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckMessage_kernal<6, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 7:
			if(Size==QCIF)
				CheckMessage_kernal<7, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckMessage_kernal<7, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 8:
			if(Size==QCIF)
				CheckMessage_kernal<8, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckMessage_kernal<8, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 9:
			if(Size==QCIF)
				CheckMessage_kernal<9, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckMessage_kernal<9, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 10:
			if(Size==QCIF)
				CheckMessage_kernal<10, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckMessage_kernal<10, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 11:
			if(Size==QCIF)
				CheckMessage_kernal<11, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckMessage_kernal<11, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 12:
			if(Size==QCIF)
				CheckMessage_kernal<12, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckMessage_kernal<12, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 13:
			if(Size==QCIF)
				CheckMessage_kernal<13, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckMessage_kernal<13, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 14:
			if(Size==QCIF)
				CheckMessage_kernal<14, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckMessage_kernal<14, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 15:
			if(Size==QCIF)
				CheckMessage_kernal<15, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckMessage_kernal<15, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 16:
			if(Size==QCIF)
				CheckMessage_kernal<16, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckMessage_kernal<16, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 17:
			if(Size==QCIF)
				CheckMessage_kernal<17, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckMessage_kernal<17, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 18:
			if(Size==QCIF)
				CheckMessage_kernal<18, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckMessage_kernal<18, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 19:
			if(Size==QCIF)
				CheckMessage_kernal<19, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckMessage_kernal<19, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 20:
			if(Size==QCIF)
				CheckMessage_kernal<20, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckMessage_kernal<20, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 21:
			if(Size==QCIF)
				CheckMessage_kernal<21, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckMessage_kernal<21, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 22:
			if(Size==QCIF)
				CheckMessage_kernal<22, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckMessage_kernal<22, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 23:
			if(Size==QCIF)
				CheckMessage_kernal<23, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckMessage_kernal<23, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 24:
			if(Size==QCIF)
				CheckMessage_kernal<24, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckMessage_kernal<24, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 25:
			if(Size==QCIF)
				CheckMessage_kernal<25, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckMessage_kernal<25, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 26:
			if(Size==QCIF)
				CheckMessage_kernal<26, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckMessage_kernal<26, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 27:
			if(Size==QCIF)
				CheckMessage_kernal<27, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckMessage_kernal<27, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 28:
			if(Size==QCIF)
				CheckMessage_kernal<28, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckMessage_kernal<28, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 29:
			if(Size==QCIF)
				CheckMessage_kernal<29, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckMessage_kernal<29, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 30:
			if(Size==QCIF)
				CheckMessage_kernal<30, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckMessage_kernal<30, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 31:
			if(Size==QCIF)
				CheckMessage_kernal<31, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckMessage_kernal<31, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 32:
			if(Size==QCIF)
				CheckMessage_kernal<32, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckMessage_kernal<32, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 33:
			if(Size==QCIF)
				CheckMessage_kernal<33, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckMessage_kernal<33, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 34:
			if(Size==QCIF)
				CheckMessage_kernal<34, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckMessage_kernal<34, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 35:
			if(Size==QCIF)
				CheckMessage_kernal<35, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckMessage_kernal<35, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 36:
			if(Size==QCIF)
				CheckMessage_kernal<36, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckMessage_kernal<36, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 37:
			if(Size==QCIF)
				CheckMessage_kernal<37, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckMessage_kernal<37, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 38:
			if(Size==QCIF)
				CheckMessage_kernal<38, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckMessage_kernal<38, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 39:
			if(Size==QCIF)
				CheckMessage_kernal<39, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckMessage_kernal<39, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 40:
			if(Size==QCIF)
				CheckMessage_kernal<40, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckMessage_kernal<40, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 41:
			if(Size==QCIF)
				CheckMessage_kernal<41, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckMessage_kernal<41, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 42:
			if(Size==QCIF)
				CheckMessage_kernal<42, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckMessage_kernal<42, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 43:
			if(Size==QCIF)
				CheckMessage_kernal<43, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckMessage_kernal<43, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 44:
			if(Size==QCIF)
				CheckMessage_kernal<44, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckMessage_kernal<44, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 45:
			if(Size==QCIF)
				CheckMessage_kernal<45, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckMessage_kernal<45, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 46:
			if(Size==QCIF)
				CheckMessage_kernal<46, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckMessage_kernal<46, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 47:
			if(Size==QCIF)
				CheckMessage_kernal<47, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckMessage_kernal<47, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 48:
			if(Size==QCIF)
				CheckMessage_kernal<48, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckMessage_kernal<48, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 49:
			if(Size==QCIF)
				CheckMessage_kernal<49, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckMessage_kernal<49, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 50:
			if(Size==QCIF)
				CheckMessage_kernal<50, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckMessage_kernal<50, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 51:
			if(Size==QCIF)
				CheckMessage_kernal<51, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckMessage_kernal<51, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 52:
			if(Size==QCIF)
				CheckMessage_kernal<52, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckMessage_kernal<52, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 53:
			if(Size==QCIF)
				CheckMessage_kernal<53, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckMessage_kernal<53, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 54:
			if(Size==QCIF)
				CheckMessage_kernal<54, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckMessage_kernal<54, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 55:
			if(Size==QCIF)
				CheckMessage_kernal<55, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckMessage_kernal<55, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 56:
			if(Size==QCIF)
				CheckMessage_kernal<56, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckMessage_kernal<56, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 57:
			if(Size==QCIF)
				CheckMessage_kernal<57, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckMessage_kernal<57, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 58:
			if(Size==QCIF)
				CheckMessage_kernal<58, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckMessage_kernal<58, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 59:
			if(Size==QCIF)
				CheckMessage_kernal<59, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckMessage_kernal<59, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 60:
			if(Size==QCIF)
				CheckMessage_kernal<60, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckMessage_kernal<60, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 61:
			if(Size==QCIF)
				CheckMessage_kernal<61, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckMessage_kernal<61, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 62:
			if(Size==QCIF)
				CheckMessage_kernal<62, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckMessage_kernal<62, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 63:
			if(Size==QCIF)
				CheckMessage_kernal<63, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckMessage_kernal<63, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>( decodedHeaderIdx,  softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
	}
}
void check_update_kernel_RT(CUstream stream, int nCode, int Size, int numBlock, int decodedHeaderIdx, Msg*  InputMessage, float*  softInputPadding, ColInfo*  Vinfo, RowInfo*  Hinfo, unsigned char*  decodedInfo){
	switch (nCode) {
		case 0:
			if(Size==QCIF)
				CheckUpdateMessage_kernal<0, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckUpdateMessage_kernal<0, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 1:
			if(Size==QCIF)
				CheckUpdateMessage_kernal<1, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckUpdateMessage_kernal<1, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 2:
			if(Size==QCIF)
				CheckUpdateMessage_kernal<2, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckUpdateMessage_kernal<2, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 3:
			if(Size==QCIF)
				CheckUpdateMessage_kernal<3, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckUpdateMessage_kernal<3, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 4:
			if(Size==QCIF)
				CheckUpdateMessage_kernal<4, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckUpdateMessage_kernal<4, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 5:
			if(Size==QCIF)
				CheckUpdateMessage_kernal<5, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckUpdateMessage_kernal<5, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 6:
			if(Size==QCIF)
				CheckUpdateMessage_kernal<6, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckUpdateMessage_kernal<6, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 7:
			if(Size==QCIF)
				CheckUpdateMessage_kernal<7, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckUpdateMessage_kernal<7, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 8:
			if(Size==QCIF)
				CheckUpdateMessage_kernal<8, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckUpdateMessage_kernal<8, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 9:
			if(Size==QCIF)
				CheckUpdateMessage_kernal<9, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckUpdateMessage_kernal<9, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 10:
			if(Size==QCIF)
				CheckUpdateMessage_kernal<10, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckUpdateMessage_kernal<10, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 11:
			if(Size==QCIF)
				CheckUpdateMessage_kernal<11, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckUpdateMessage_kernal<11, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 12:
			if(Size==QCIF)
				CheckUpdateMessage_kernal<12, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckUpdateMessage_kernal<12, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 13:
			if(Size==QCIF)
				CheckUpdateMessage_kernal<13, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckUpdateMessage_kernal<13, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 14:
			if(Size==QCIF)
				CheckUpdateMessage_kernal<14, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckUpdateMessage_kernal<14, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 15:
			if(Size==QCIF)
				CheckUpdateMessage_kernal<15, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckUpdateMessage_kernal<15, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 16:
			if(Size==QCIF)
				CheckUpdateMessage_kernal<16, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckUpdateMessage_kernal<16, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 17:
			if(Size==QCIF)
				CheckUpdateMessage_kernal<17, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckUpdateMessage_kernal<17, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 18:
			if(Size==QCIF)
				CheckUpdateMessage_kernal<18, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckUpdateMessage_kernal<18, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 19:
			if(Size==QCIF)
				CheckUpdateMessage_kernal<19, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckUpdateMessage_kernal<19, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 20:
			if(Size==QCIF)
				CheckUpdateMessage_kernal<20, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckUpdateMessage_kernal<20, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 21:
			if(Size==QCIF)
				CheckUpdateMessage_kernal<21, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckUpdateMessage_kernal<21, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 22:
			if(Size==QCIF)
				CheckUpdateMessage_kernal<22, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckUpdateMessage_kernal<22, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 23:
			if(Size==QCIF)
				CheckUpdateMessage_kernal<23, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckUpdateMessage_kernal<23, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 24:
			if(Size==QCIF)
				CheckUpdateMessage_kernal<24, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckUpdateMessage_kernal<24, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 25:
			if(Size==QCIF)
				CheckUpdateMessage_kernal<25, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckUpdateMessage_kernal<25, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 26:
			if(Size==QCIF)
				CheckUpdateMessage_kernal<26, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckUpdateMessage_kernal<26, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 27:
			if(Size==QCIF)
				CheckUpdateMessage_kernal<27, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckUpdateMessage_kernal<27, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 28:
			if(Size==QCIF)
				CheckUpdateMessage_kernal<28, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckUpdateMessage_kernal<28, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 29:
			if(Size==QCIF)
				CheckUpdateMessage_kernal<29, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckUpdateMessage_kernal<29, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 30:
			if(Size==QCIF)
				CheckUpdateMessage_kernal<30, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckUpdateMessage_kernal<30, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 31:
			if(Size==QCIF)
				CheckUpdateMessage_kernal<31, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckUpdateMessage_kernal<31, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 32:
			if(Size==QCIF)
				CheckUpdateMessage_kernal<32, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckUpdateMessage_kernal<32, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 33:
			if(Size==QCIF)
				CheckUpdateMessage_kernal<33, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckUpdateMessage_kernal<33, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 34:
			if(Size==QCIF)
				CheckUpdateMessage_kernal<34, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckUpdateMessage_kernal<34, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 35:
			if(Size==QCIF)
				CheckUpdateMessage_kernal<35, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckUpdateMessage_kernal<35, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 36:
			if(Size==QCIF)
				CheckUpdateMessage_kernal<36, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckUpdateMessage_kernal<36, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 37:
			if(Size==QCIF)
				CheckUpdateMessage_kernal<37, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckUpdateMessage_kernal<37, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 38:
			if(Size==QCIF)
				CheckUpdateMessage_kernal<38, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckUpdateMessage_kernal<38, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 39:
			if(Size==QCIF)
				CheckUpdateMessage_kernal<39, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckUpdateMessage_kernal<39, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 40:
			if(Size==QCIF)
				CheckUpdateMessage_kernal<40, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckUpdateMessage_kernal<40, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 41:
			if(Size==QCIF)
				CheckUpdateMessage_kernal<41, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckUpdateMessage_kernal<41, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 42:
			if(Size==QCIF)
				CheckUpdateMessage_kernal<42, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckUpdateMessage_kernal<42, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 43:
			if(Size==QCIF)
				CheckUpdateMessage_kernal<43, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckUpdateMessage_kernal<43, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 44:
			if(Size==QCIF)
				CheckUpdateMessage_kernal<44, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckUpdateMessage_kernal<44, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 45:
			if(Size==QCIF)
				CheckUpdateMessage_kernal<45, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckUpdateMessage_kernal<45, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 46:
			if(Size==QCIF)
				CheckUpdateMessage_kernal<46, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckUpdateMessage_kernal<46, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 47:
			if(Size==QCIF)
				CheckUpdateMessage_kernal<47, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckUpdateMessage_kernal<47, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 48:
			if(Size==QCIF)
				CheckUpdateMessage_kernal<48, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckUpdateMessage_kernal<48, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 49:
			if(Size==QCIF)
				CheckUpdateMessage_kernal<49, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckUpdateMessage_kernal<49, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 50:
			if(Size==QCIF)
				CheckUpdateMessage_kernal<50, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckUpdateMessage_kernal<50, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 51:
			if(Size==QCIF)
				CheckUpdateMessage_kernal<51, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckUpdateMessage_kernal<51, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 52:
			if(Size==QCIF)
				CheckUpdateMessage_kernal<52, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckUpdateMessage_kernal<52, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 53:
			if(Size==QCIF)
				CheckUpdateMessage_kernal<53, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckUpdateMessage_kernal<53, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 54:
			if(Size==QCIF)
				CheckUpdateMessage_kernal<54, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckUpdateMessage_kernal<54, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 55:
			if(Size==QCIF)
				CheckUpdateMessage_kernal<55, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckUpdateMessage_kernal<55, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 56:
			if(Size==QCIF)
				CheckUpdateMessage_kernal<56, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckUpdateMessage_kernal<56, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 57:
			if(Size==QCIF)
				CheckUpdateMessage_kernal<57, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckUpdateMessage_kernal<57, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 58:
			if(Size==QCIF)
				CheckUpdateMessage_kernal<58, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckUpdateMessage_kernal<58, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 59:
			if(Size==QCIF)
				CheckUpdateMessage_kernal<59, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckUpdateMessage_kernal<59, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 60:
			if(Size==QCIF)
				CheckUpdateMessage_kernal<60, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckUpdateMessage_kernal<60, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 61:
			if(Size==QCIF)
				CheckUpdateMessage_kernal<61, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckUpdateMessage_kernal<61, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 62:
			if(Size==QCIF)
				CheckUpdateMessage_kernal<62, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckUpdateMessage_kernal<62, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
		case 63:
			if(Size==QCIF)
				CheckUpdateMessage_kernal<63, QCIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			else
				CheckUpdateMessage_kernal<63, CIF><<<numBlock, LDPCA_BLOCK_SIZE, 0, (cudaStream_t)(stream)>>>(decodedHeaderIdx, InputMessage, softInputPadding, Vinfo, Hinfo, decodedInfo);
			break;
	}
}

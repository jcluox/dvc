#include <stdio.h>
#include <stdlib.h>

//#include "defines.h"
#include "global.h"
#include "error.h"
#include "LdpcaDecoder.h"
#include "ReadLdpcaLadderFile_LdpcaDecoder.h"

//cuda
#ifdef CUDA
#include "LDPCA_cuda.h"
#include "updateSI_kernel.h"
#include "forwardME_kernel.h"
#include "motionLearning_kernel.h"
#include "noiseModel_kernel.h"
#endif

#include "transform.h"
#include "reconstruction.h"
#include "frame.h"
#include "DVC_Decoder.h"
#include "configfile.h"
#include "wzbitstream.h"
#include "sideInfoCreation.h"
#include "intraBitstream.h"

int main(int argc, char** argv){

	#ifdef _WIN32
		LARGE_INTEGER m_liPerfFreqLDPCA={0};
		QueryPerformanceFrequency(&m_liPerfFreqLDPCA);
		CPUFreq = (double)(m_liPerfFreqLDPCA.QuadPart);
	#endif

	//read configure file
	printf("Read configure file...\n");
	readConfigure(argc, argv);

	//read WZ bitstream file
	printf("Read WZ bitstream file...\n");
	readWZBitstream(codingOption.WZBitStreamFile);
	displayDecodeParamters();

	//read Intra bitstream file
	if(codingOption.IntraModeExist){
		printf("Read Intra bitstream file...\n");
		readIntraBitstream(codingOption.IntraBitStreamFile);
	}

	initFrameBuffer();
	init_1616ME(SearchInfo1616);

	//read input yuv
	printf("Read original sequence...\n");
	readSequence(codingOption.OriYuvFile, oriFrames);
	//read reconstruct yuv
	printf("Read key frame sequence...\n");
	unsigned char **keyFrames = (unsigned char**)malloc(sizeof(unsigned char*) * codingOption.FramesToBeDecoded);
	for(int i=0; i<codingOption.FramesToBeDecoded; i++)
		keyFrames[i] = (unsigned char*)malloc(sizeof(unsigned char) * FrameSize);
	readSequence(codingOption.KeyFrameYuvFile, keyFrames);
	for(int i=0; i<codingOption.FramesToBeDecoded; i++){
		for(int j=0; j<FrameSize; j++){
			decodedFrames[i].reconsFrame[j] = keyFrames[i][j];
		}
		decodedFrames[i].hasReferenced = 0;
		decodedFrames[i].isWZFrame = 0;
		free(keyFrames[i]);
	}
	free(keyFrames);

	//read key frame information
	printf("Read key frame information...\n");
	for(int i=0; i<codingOption.FramesToBeDecoded; i++)
		read_one_frame_info(i, &(decodedFrames[i].bitRate), &(decodedFrames[i].PSNR), codingOption.KeyFrameInfo);

	//read Ladder file
	char LadderFileName[50];
	char pInverseHmatrixFileName[50];
	printf("Read Ladder file...\n");
	sprintf(LadderFileName, "%s", (codingOption.FrameWidth == 176) ? "LDPCA_LadderFile/RegDecoder1584.lad" : "LDPCA_LadderFile/RegDecoder6336.lad");
	sprintf(pInverseHmatrixFileName, "%s", (codingOption.FrameWidth == 176) ? "LDPCA_LadderFile/Inverse_Matrix_H_Reg1584.bin" : "LDPCA_LadderFile/Inverse_Matrix_H_Reg6336.bin");
	//sprintf(LadderFileName, "%s", (codingOption.FrameWidth == 176) ? "LDPCA_LadderFile/irReg1584.lad" : "LDPCA_LadderFile/RegDecoder6336.lad");  //JSRF irregular	
	//sprintf(pInverseHmatrixFileName, "%s", (codingOption.FrameWidth == 176) ? "LDPCA_LadderFile/Inverse_Matrix_H_irReg1584.dat" : "LDPCA_LadderFile/Inverse_Matrix_H_Reg6336.dat");

	printf("Ladder file name: %s\n", LadderFileName);
	printf("Inverse H matrix file name: %s\n", pInverseHmatrixFileName);
	ReadLadder_decoder(LadderFileName, pInverseHmatrixFileName);	

	#ifdef CUDA
		if(codingOption.ParallelMode > 1){
			//CUDA
			if(InitCUDA() == false){
				getchar();
				exit(-1);
			}		
			LDPCA_CUDA_Init_Variable();
			ME_CUDA_Init_Buffer();
			CNM_CUDA_Init_Buffer();
			ForwardME_CUDA_Init_Buffer();
			updateSMF_CUDA_Init_Buffer();
		}
	#endif

	//WZ decode
	printf("Start decoding...\n\n");
					
	TimeStamp st_time, end_time;
	timeStart(&st_time);

	initTransform();
	initLdpcaBuffer();
	#ifdef SHOW_TIME_INFO
		printf("FrameNo: 0, Y_bits: %d bits, PSNR: %.2lf (key frame)\n\n", decodedFrames[0].bitRate, decodedFrames[0].PSNR);
	#endif

	if(codingOption.startFrame == -1 || codingOption.endFrame == -1 ){  //decode the whole sequence		
		codingOption.startFrame = 0;
		codingOption.endFrame = codingOption.FramesToBeDecoded-1;
	}
	else{  //把intra bitstream在startFrame以前的資料全部讀掉
		for(int i=0; i<codingOption.startFrame; i+=codingOption.GOP){
			//for each GOP
			int begin = i;
			int end = i + codingOption.GOP;
			if(end > codingOption.endFrame)
				end = codingOption.endFrame;
			skipGOP(begin, end);
		}
	}
	for(int i=codingOption.startFrame; i<=codingOption.endFrame; i+=codingOption.GOP){
		//for each GOP
		int begin = i;
		int end = i + codingOption.GOP;
		if(end > codingOption.endFrame)
			end = codingOption.endFrame;
		#ifdef SHOW_TIME_INFO
			printf("FrameNo: %d, Y_bits: %d bits, PSNR: %.2lf (key frame)\n\n", end, decodedFrames[end].bitRate, decodedFrames[end].PSNR);
		#endif
		decodeGOP(begin, end);
	}

	//output total bits, PSNR, decoding time
	double totalDecodeTime = timeElapse(st_time,&end_time)/1000.0;	

	printf("output decoded RD file: %s\n", codingOption.RDFile);
	reportDecodeInfo(codingOption.RDFile, totalDecodeTime);

	//output decoded sequence (only y)
	printf("output decoded yuv file (only y): %s\n", codingOption.OutputDecodedYuvFile);

	outputSequence(codingOption.OutputDecodedYuvFile);
	//release resource
	#ifdef CUDA
		if(codingOption.ParallelMode > 1){				
			ME_CUDA_Free_Buffer(); 
			CNM_CUDA_Free_Buffer();
			ForwardME_CUDA_Free_Buffer();
			updateSMF_CUDA_Free_Buffer();
			LDPCA_CUDA_Free_Variable();
		}
	#endif
	freeLdpcaBuffer();
	freeLdpcaParam();
	freeFrameBuffer();
	if(codingOption.IntraModeExist)
		free_IntraBuffer();

	return 0;
}


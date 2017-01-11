#include <stdio.h>
#include <stdlib.h>

#ifdef LINUX
#include <sys/time.h>
#else
#include <windows.h>
#endif

#include "global.h"
#include "error.h"
#include "ReadLdpcaLadderFile_LdpcaEncoder.h"
#include "quantization.h"
#include "transform.h"
#include "DVC_Encoder.h"
#include "frame.h"
#include "configfile.h"
#include "wzbitstream.h"
#include "intraBitstream.h"
#include "modeSelection.h"

int main(int argc, char** argv){

#ifndef LINUX
	LARGE_INTEGER m_liPerfFreqLDPCA={0};
	QueryPerformanceFrequency(&m_liPerfFreqLDPCA);
	CPUFreq = (double)(m_liPerfFreqLDPCA.QuadPart);
#endif

	//read configure file
	printf("Read configure file...\n");
	readConfigure(argc, argv);

	//read input yuv
	printf("Read input sequence...\n");
	initFrameBuffer();
	readSequence(codingOption.InputFile, oriFrames);
	//test
	//saveToBmp(oriFrames[1], codingOption.FrameWidth, codingOption.FrameHeight, "oriFrames1.bmp");

	if(codingOption.EnableIntraMode){
		//read H264 reconstruct yuv
		printf("Read key frame sequence...\n");
		readSequence(codingOption.KeyFrameYuvFile, reconsFrames);

		//open output file of intra block bitstream
		IntraFp = fopen(codingOption.IntraOutputFile, "wb+");
		if(IntraFp == NULL){
			sprintf(errorText, "File %s doesn't exist\n", codingOption.IntraOutputFile);
			errorMsg(errorText);
		}

		intrap.thita = 0.0;
        intrap.kappa = 0.0;
        init_IntraBuffer();
	}
	
	//read Ladder file
	char LadderFileName[50];
	printf("Read Ladder file...\n");
	sprintf(LadderFileName, "%s", (codingOption.FrameWidth == 176)?"LDPCA_LadderFile/RegEncoder1584.lad" : "LDPCA_LadderFile/RegEncoder6336.lad");
	//sprintf(LadderFileName, "%s", (codingOption.FrameWidth == 176) ? "LDPCA_LadderFile/irRegEncoder1584.lad" : "LDPCA_LadderFile/RegDecoder6336.lad");  //JSRF irregular
	printf("Ladder file name: %s\n", LadderFileName);
	ReadLadder_encoder(LadderFileName);

	//WZ encode
	printf("WZ encode...\n");
#ifdef LINUX	
	struct timeval st_time, end_time;
    gettimeofday(&st_time, NULL);
#else
	LARGE_INTEGER st_time={0}, end_time={0};
	QueryPerformanceCounter(&st_time);
#endif
	//SkipMin = (int)((double)Block88Num * U / (double)QbitsNumPerBlk);
	initTransform();
	int wzCount = 0;
	for(int i=0; i<codingOption.FramesToBeEncoded; i+=codingOption.GOP){
		//for each GOP
		int begin = i;
		int end = i + codingOption.GOP;
		if(end >= codingOption.FramesToBeEncoded)
			end = codingOption.FramesToBeEncoded - 1;
		encodeGOP(begin, end, &wzCount);
	}
#ifdef LINUX
	gettimeofday(&end_time, NULL);    
	double totalEncodeTime = (double)(end_time.tv_sec - st_time.tv_sec)*1000.0 + (double)(end_time.tv_usec - st_time.tv_usec)/1000.0;  //millisecond
#else
	QueryPerformanceCounter(&end_time);
	double totalEncodeTime = (double)(end_time.QuadPart - st_time.QuadPart)*1000.0/CPUFreq;  //millisecond
#endif
	printf("\nTotal encoding time: %.2lfms\nAeverage: %.2lfms per frame\n", totalEncodeTime, totalEncodeTime/(double)wzCount);

	//output WZ bitstream
	writeWZbitstream(codingOption.WZOutputFile);
	if(codingOption.EnableIntraMode){
		//output remaining intra bitstream
        free_IntraBuffer();
    }

	//release resource
	freeLdpcaParam();
	freeFrameBuffer();
	
	return 0;
}


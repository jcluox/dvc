#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "global.h"
#include "error.h"
#include "quantization.h"

void initFrameBuffer(){
	oriFrames = (unsigned char**)malloc(sizeof(unsigned char*) * codingOption.FramesToBeEncoded);
	for(int i=0; i<codingOption.FramesToBeEncoded; i++)
		oriFrames[i] = (unsigned char*)malloc(sizeof(unsigned char) * FrameSize);

	reconsFrames = (unsigned char**)malloc(sizeof(unsigned char*) * codingOption.FramesToBeEncoded);
	for(int i=0; i<codingOption.FramesToBeEncoded; i++)
		reconsFrames[i] = (unsigned char*)malloc(sizeof(unsigned char) * FrameSize);
	
	encodedFrames = (EncodedFrame*)malloc(sizeof(EncodedFrame) * codingOption.FramesToBeEncoded);
	for(int i=0; i<codingOption.FramesToBeEncoded; i++){
		encodedFrames[i].skipBlockFlag = (unsigned char*)malloc(sizeof(unsigned char) * Block88Num);
		encodedFrames[i].intraBlockFlag= (unsigned char*)malloc(sizeof(unsigned char) * Block88Num);
		encodedFrames[i].CRC = (unsigned char*)malloc(sizeof(unsigned char) * QbitsNumPerBlk);
		encodedFrames[i].AccumulatedSyndrome = (unsigned char**)malloc(sizeof(unsigned char*) * QbitsNumPerBlk);
		encodedFrames[i].isWZ = 0;
		encodedFrames[i].intraCount = 0;
		encodedFrames[i].E = (double*)malloc(sizeof(double) * Block44Num);
        encodedFrames[i].Z = (double*)malloc(sizeof(double) * Block44Num);
        encodedFrames[i].Rspent = (int*)malloc(sizeof(int) * Block44Num);
        encodedFrames[i].blockInfo = (BlockData*)malloc(sizeof(BlockData) * Block44Num);
        memset(encodedFrames[i].blockInfo, 0, sizeof(BlockData) * Block44Num);
        memset(encodedFrames[i].E, 0, sizeof(double) * Block44Num);
        memset(encodedFrames[i].Z, 0, sizeof(double) * Block44Num);
        memset(encodedFrames[i].Rspent, 0, sizeof(int) * Block44Num);
	}
	alpha_perframe = (double *)malloc(sizeof(double)*FrameSize);
}

void freeFrameBuffer(){
	for(int i=0; i<codingOption.FramesToBeEncoded; i++){
		free(oriFrames[i]);
		free(reconsFrames[i]);
	}
	free(oriFrames);
	free(reconsFrames);

	for(int i=0; i<codingOption.FramesToBeEncoded; i++){
		if(encodedFrames[i].isWZ == 1){
		for(int j=0; j<QbitsNumPerBlk; j++)
			free(encodedFrames[i].AccumulatedSyndrome[j]);
		}
		free(encodedFrames[i].skipBlockFlag);
		free(encodedFrames[i].intraBlockFlag);
		free(encodedFrames[i].AccumulatedSyndrome);
		free(encodedFrames[i].CRC);
		free(encodedFrames[i].E);
        free(encodedFrames[i].Z);
        free(encodedFrames[i].Rspent);
        free(encodedFrames[i].blockInfo);
	}
	free(encodedFrames);
    free(Kmatrix);
    free(alpha_perframe);
}

void readSequence(char *fileName, unsigned char **frames){
	FILE *fp = fopen(fileName, "rb");
	if(fp == NULL){
		sprintf(errorText, "File %s doesn't exist\n", fileName);
		errorMsg(errorText);
	}

	int uvSize = FrameSize/2;
	unsigned char *buf = (unsigned char *)malloc(sizeof(unsigned char) * uvSize); 
	for(int i=0; i<codingOption.FramesToBeEncoded; i++){
		//read y
		if(fread(frames[i], 1, FrameSize, fp) != FrameSize){
			sprintf(errorText, "Input sequence %s is not enough", fileName);
			errorMsg(errorText);
		}
		
		//read u,v
		fread(buf, 1, uvSize, fp);
	}	
	free(buf);
	fclose(fp);	
}

void write_bmpheader(unsigned char *bitmap, int offset, int bytes, int value){
	int i;
	for(i=0; i<bytes; i++)
		bitmap[offset+i] = (value >> (i<<3)) & 0xFF;
}

unsigned char *convertToBmp(unsigned char *inputImg, int width, int height, int *ouputSize){
	/*create a bmp format file*/
	int bitmap_x = (int)ceil((double)width*3/4) * 4;	
	unsigned char *bitmap = (unsigned char*)malloc(sizeof(unsigned char)*height*bitmap_x + 54);
	
	bitmap[0] = 'B';
	bitmap[1] = 'M';
	write_bmpheader(bitmap, 2, 4, height*bitmap_x+54); //whole file size
	write_bmpheader(bitmap, 0xA, 4, 54); //offset before bitmap raw data
	write_bmpheader(bitmap, 0xE, 4, 40); //length of bitmap info header
	write_bmpheader(bitmap, 0x12, 4, width); //width
	write_bmpheader(bitmap, 0x16, 4, height); //height
	write_bmpheader(bitmap, 0x1A, 2, 1);
	write_bmpheader(bitmap, 0x1C, 2, 24); //bit per pixel
	write_bmpheader(bitmap, 0x1E, 4, 0); //compression
	write_bmpheader(bitmap, 0x22, 4, height*bitmap_x); //size of bitmap raw data
	for(int i=0x26; i<0x36; i++)
		bitmap[i] = 0;

	int k=54;
	for(int i=height-1; i>=0; i--){
		int j;
		for(j=0; j<width; j++){
			int index = i*width+j;
			for(int l=0; l<3; l++)
				bitmap[k++] = inputImg[index];
		}
		j*=3;
		while(j<bitmap_x){
			bitmap[k++] = 0;
			j++;
		}
	}
	
	*ouputSize = k;
	return bitmap;
}

void saveToBmp(unsigned char *inputImg, int width, int height, char *outputFileName){
	int size;
	unsigned char *bmp = convertToBmp(inputImg, width, height, &size);
	FILE *fp = fopen(outputFileName, "wb+");
	fwrite(bmp, 1, size, fp);
	fclose(fp);
	free(bmp);
}

void printBlock44(int *blk){
	for(int i=0; i<4; i++){
		for(int j=0; j<4; j++)
			printf("%4d", blk[i*4+j]);
		printf("\n");
	}
#ifndef LINUX
	system("pause");
#endif
}
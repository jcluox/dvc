#include <stdio.h>
#include <stdlib.h>

#include "global.h"
#include "intraBitstream.h"
#include "error.h"

Bitstream IntraBS;

void readIntraBitstream(char *fileName){
	FILE *fp = fopen(fileName, "rb");
	if(fp == NULL){
		sprintf(errorText, "File %s doesn't exist", fileName);
		errorMsg(errorText);
	}
	printf("Intra bitstream file name: %s\n", fileName);

	init_IntraBuffer(fp);
	fread(IntraBS.streamBuffer, sizeof(unsigned char), IntraBS.bitstream_length, fp);
	fclose(fp);	
}

void init_IntraBuffer(FILE *fp){
    unsigned int size;
    fseek(fp, 0, SEEK_END);
    size = ftell(fp);
    
    rewind(fp);
    IntraBS.streamBuffer = (unsigned char*)malloc(sizeof(unsigned char) * size);
    IntraBS.bitstream_length = size;
    IntraBS.frame_bitoffset = 0;
    IntraBS.ei_flag = 0;
    IntraBS.prev_frame_bitoffset = 0;
}


void free_IntraBuffer(){
    free(IntraBS.streamBuffer);
}

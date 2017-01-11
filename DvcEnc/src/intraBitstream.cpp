#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "intraBitstream.h"
#include "sJBIG.h"

void printbinary(int pattern,int len){
    int *out;
    out = (int*)malloc(sizeof(int)*len);

    for(int i=0;i<len;i++){
        out[i] = pattern & 1;
        pattern >>= 1;
    }
    for(int i=len-1;i>=0;i--){
        printf("%d",out[i]);
    }
    //printf("\n");

    free(out);
}

void init_IntraBuffer(){

    Bitstream *currStream;

    currStream = &IntraBuffer;
    currStream->bits_to_go = 8;
    currStream->byte_pos = 0;    
    currStream->byte_buf = 0;
    currStream->buffer_size = FrameSize;
    //currStream->write_flag = 0;
    currStream->streamBuffer = (unsigned char*)malloc(sizeof(unsigned char) * FrameSize);
    
    //printf("frame buffer byte = %d\n",FrameSize);
}

/*unsigned char *BitToBitmap(unsigned char *Bits, int width, int height, int *numBytes){
    int remainder;
    int WidthByteLength;
    unsigned char *Bytes;


    WidthByteLength = width >> 3; //BitLength / 8
    remainder = width & 7;  //BitLength % 8
    if(remainder != 0)
        WidthByteLength++;

    Bytes = (unsigned char *)malloc(sizeof(unsigned char) * WidthByteLength * height);
    *numBytes =  WidthByteLength * height; //output number of bytes
    
    int byteIdx = 0;
    unsigned char tmp = 0;
    for(int j=0;j<height;j++){
        tmp = 0;
        for(int i=0; i<width; i++){
            tmp |= Bits[i+j*width];
            if(((i+1)&7) == 0){ //(i+1)%8 == 0
                Bytes[byteIdx++] = tmp;
                tmp = 0;
            }
            else
                tmp <<= 1;
        }
        
        if(remainder != 0)
            Bytes[byteIdx++] = tmp<<(7-remainder);
            
    }

    return Bytes;
}*/


//write remaining intra bitstream to file
void free_IntraBuffer(){
    if(IntraBuffer.bits_to_go != 8){
        unsigned char temp = IntraBuffer.byte_buf;
        temp <<= IntraBuffer.bits_to_go;
        //printf("last byte: ");
        //printbinary(temp,8);
        fwrite(&temp, sizeof(unsigned char), 1, IntraFp);
    }
    free(IntraBuffer.streamBuffer);
	fclose(IntraFp);
}

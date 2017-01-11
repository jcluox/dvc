#ifndef FRAME_H
#define FRAME_H

void initFrameBuffer();
void freeFrameBuffer();
void readSequence(char *fileName, unsigned char **frames);
void write_bmpheader(unsigned char *bitmap, int offset, int bytes, int value);
unsigned char *convertToBmp(unsigned char *inputImg, int width, int height, int *ouputSize);
void saveToBmp(unsigned char *inputImg, int width, int height, char *outputFileName);
void printBlock44(int *blk);

#endif
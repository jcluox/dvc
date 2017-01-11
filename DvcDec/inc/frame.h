#ifndef FRAME_H
#define FRAME_H

void initFrameBuffer();
void freeFrameBuffer();
void readSequence(char *fileName, unsigned char **frames);
void outputSequence(char *fileName);

double psnr(unsigned char *frame1, unsigned char *frame2);
void write_bmpheader(unsigned char *bitmap, int offset, int bytes, int value);

unsigned char *convertToBmp(unsigned char *inputImg, int width, int height, int *ouputSize);
void saveToBmp(unsigned char *inputImg, int width, int height, char *outputFileName);
void printBlock44(int *blk);

void read_one_frame_info(int frameNo, int *y_bits, double *y_psnr, char *fileName);

void reportDecodeInfo(char *fileName, double totalDecodeTime);

#endif

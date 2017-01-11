#ifndef SJBIG_H
#define SJBIG_H

unsigned char *BitToBitmap(unsigned char *Bits, int width, int height, int *numBytes);
int simple_JBIG_enc(FILE *fp,unsigned char *jbigdata,int width, int height, unsigned char jbgh[20], int stripe, int write);

#endif
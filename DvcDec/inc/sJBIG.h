#ifndef SJBIG_H
#define SJBIG_H

unsigned char* ByteToBitmap(unsigned char *Bytes, int width,int height);
int simple_JBIG_dec(unsigned char *bdm_data, long int bdm_rsize,unsigned char jbgh[20], int total_len,unsigned char *bdm_buffer);

#endif
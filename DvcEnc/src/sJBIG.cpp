#include <stdio.h>
#include <stdlib.h>
#include "jbig.h"

int total;

void output_bie(unsigned char *start, size_t len, void *file){
    fwrite(start, 1, len, (FILE *) file);
    total += (unsigned long)len;
}

void output_bie_fake(unsigned char *start, size_t len, void *file){
    total += (unsigned long)len;
    start = start;
    file = file;
}

unsigned char *BitToBitmap(unsigned char *Bits, int width, int height, int *numBytes){
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
}

int simple_JBIG_enc(FILE *fp,unsigned char *jbigdata,int width, int height, unsigned char jbgh[20], int stripe, int write){
    total = 0;
    unsigned char *bitmaps[1] = { jbigdata };
    struct jbg_enc_state se;

    if(write)
        jbg_enc_init(&se, width, height, 1, bitmaps, output_bie, fp);
    else
        jbg_enc_init(&se, width, height, 1, bitmaps, output_bie_fake, fp);

    //int stripe = 36*bdm_num;
    jbg_enc_options(&se, JBG_ILEAVE | JBG_SMID, JBG_TPDON | JBG_DPON ,stripe, 8, 0);

    jbg_enc_out(&se,jbgh);
    jbg_enc_free(&se);

    return total;
}

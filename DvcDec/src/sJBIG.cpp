#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "error.h"
#include "jbig.h"


void printbinary2(int pattern,int len){
    int *out;
    out = (int*)malloc(sizeof(int)*len);

    for(int i=0;i<len;i++){
        out[i] = pattern & 1;
        pattern >>= 1;
    }
    for(int i=len-1;i>=0;i--){
        printf("%d",out[i]);
    }

    free(out);
}

unsigned char* ByteToBitmap(unsigned char *Bytes, int width,int height){
    unsigned char *Bits, tmp=0;
    int width_byte = ((width + 7) >> 3) << 3;


    if((width_byte*height) < (width*height))
        errorMsg("In function ByteToBit: SourceLength is larger than ByteLength");

    Bits = (unsigned char *)malloc(sizeof(unsigned char) * width * height);

    int byteIdx = 0;
    for(int i=0;i<height;i++){
        tmp = 0;
        for(int j=0;j<width;j++){
            if((j&7) == 0)
                tmp = Bytes[byteIdx++];
            else
                tmp <<= 1;
            Bits[i*width+j] = tmp >> 7;
        }
    }

    return Bits;
}

int simple_JBIG_dec(unsigned char *bdm_data, long int bdm_rsize,unsigned char jbgh[20], int total_len,unsigned char *bdm_buffer){
    struct jbg_dec_state sd;
    int buffer_size = (total_len+20);
    unsigned char *data = (unsigned char*)malloc(sizeof(unsigned char)*buffer_size);
    size_t real_byte;
    int temp;

    jbg_dec_init(&sd);


    memcpy(data,jbgh,sizeof(unsigned char)*20);
    
    if((long int)total_len > bdm_rsize){
        memcpy(&data[20],bdm_data,sizeof(unsigned char)*bdm_rsize);
        //printf("copy %ld bytes\n",bdm_rsize);
    }else{
        memcpy(&data[20],bdm_data,sizeof(unsigned char)*total_len);
        //printf("copy %d bytes\n",total_len);
    }

    //temp = fread(&data[20],sizeof(unsigned char),total_len,fp);    


    unsigned char *data_temp = data;
    int decode_byte = 0;
    while(1){
        temp = jbg_dec_in(&sd, data_temp, buffer_size,&real_byte);
        //printf("return = %d, decode %d bytes\n",temp,(int)real_byte);
        decode_byte += (int)real_byte;
        if(temp == JBG_EOK_INTR){
            //printf("JBG_EOK_INTR\n");
            break;
        }else if(temp == JBG_EOK){
            //printf("JBG_EOK\n");
            break;
        }else if(temp == JBG_EAGAIN){
            printf("JBG_EAGAIN\n");
            data_temp = &data[real_byte];
        }else{
            errorMsg("JBG decoder.");    
        }
    }

    unsigned char *decode_bitmap = jbg_dec_getimage(&sd, 0);
    //printf("width = %ld\n",jbg_dec_getwidth(&sd));
    //printf("height = %ld\n",jbg_dec_getheight(&sd));
    //printf("size = %ld\n",jbg_dec_getsize(&sd));


    memcpy(bdm_buffer,decode_bitmap,sizeof(unsigned char)*jbg_dec_getsize(&sd));

    free(data);
    jbg_dec_free(&sd);
    return (decode_byte - 20);
}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vlc.h"
#include "intraBitstream.h"
#include "inlineFunc.h" 
#include "global.h"

#define SE_LUM_AC_INTRA 9
#define RUNBEFORE_NUM_M1 6
#define TOTRUN_NUM 15

/*!
 ************************************************************************
 * \brief
 *  Reads bits from the bitstream buffer (Threshold based)
 *
 * \param inf
 *    bytes to extract numbits from with bitoffset already applied
 * \param numbits
 *    number of bits to read
 *
 ************************************************************************
 */
int ShowBitsThres (int inf, int numbits)
{
  return ((inf) >> ((sizeof(unsigned char) * 24) - (numbits)));
}

/*!
 ************************************************************************
 * \brief
 *    code from bitstream (2d tables)
 ************************************************************************
 */
int code_from_bitstream_2d(SyntaxElement *sym, Bitstream *currStream, 
        const unsigned char *lentab,
        const unsigned char *codtab,
        int tabwidth,
        int tabheight,
        int *code)
{
    int i, j;
    const unsigned char *len = &lentab[0], *cod = &codtab[0];

    unsigned long int *frame_bitoffset = &currStream->frame_bitoffset;
    unsigned char *buf            = &currStream->streamBuffer[*frame_bitoffset >> 3];

    //Apply bitoffset to three bytes (maximum that may be traversed by ShowBitsThres)
    unsigned int inf = ((*buf) << 16) + (*(buf + 1) << 8) + *(buf + 2); 
    //Even at the end of a stream we will still be pulling out of allocated memory as alloc is done by MAX_CODED_FRAME_SIZE
    inf <<= (*frame_bitoffset & 0x07);                                  
    //Offset is constant so apply before extracting different numbers of bits
    inf  &= 0xFFFFFF;                                                   
    //Arithmetic shift so wipe any sign which may be extended inside ShowBitsThres

    // this VLC decoding method is not optimized for speed
    for (j = 0; j < tabheight; j++) 
    {
        for (i = 0; i < tabwidth; i++)
        {
            if ((*len == 0) || (ShowBitsThres(inf, (int) *len) != *cod))
            {
                len++;
                cod++;
            }
            else
            {
                sym->len = *len;
                *frame_bitoffset += *len; // move bitstream pointer
                *code = *cod;             
                sym->value1 = i;
                sym->value2 = j;        
                return 0;                 // found code and return 
            }
        }
    }
    return -1;  // failed to find code
}


/*!
 ************************************************************************
 * \brief
 *    read  Run codeword from UVLC-partition
 ************************************************************************
 */
int readSyntaxElement_Run(SyntaxElement *sym, Bitstream *currStream){
    static const unsigned char lentab[TOTRUN_NUM][16] =
    {
        {1,1},
        {1,2,2},
        {2,2,2,2},
        {2,2,2,3,3},
        {2,2,3,3,3,3},
        {2,3,3,3,3,3,3},
        {3,3,3,3,3,3,3,4,5,6,7,8,9,10,11},
    };

    static const unsigned char codtab[TOTRUN_NUM][16] =
    {
        {1,0},
        {1,1,0},
        {3,2,1,0},
        {3,2,1,1,0},
        {3,2,3,2,1,0},
        {3,0,1,3,2,5,4},
        {7,6,5,4,3,2,1,1,1,1,1,1,1,1,1},
    };
    int code;
    int vlcnum = sym->value1;
    int retval = code_from_bitstream_2d(sym, currStream, &lentab[vlcnum][0], &codtab[vlcnum][0], 16, 1, &code);

    if (retval)
    {
        printf("ERROR: failed to find Run\n");
        exit(-1);
    }

    return retval;
}

/*!
 ************************************************************************
 * \brief
 *    read Total Zeros codeword from UVLC-partition
 ************************************************************************
 */
int readSyntaxElement_TotalZeros(SyntaxElement *sym,  Bitstream *currStream){
    static const unsigned char lentab[TOTRUN_NUM][16] =
    {

        { 1,3,3,4,4,5,5,6,6,7,7,8,8,9,9,9},
        { 3,3,3,3,3,4,4,4,4,5,5,6,6,6,6},
        { 4,3,3,3,4,4,3,3,4,5,5,6,5,6},
        { 5,3,4,4,3,3,3,4,3,4,5,5,5},
        { 4,4,4,3,3,3,3,3,4,5,4,5},
        { 6,5,3,3,3,3,3,3,4,3,6},
        { 6,5,3,3,3,2,3,4,3,6},
        { 6,4,5,3,2,2,3,3,6},
        { 6,6,4,2,2,3,2,5},
        { 5,5,3,2,2,2,4},
        { 4,4,3,3,1,3},
        { 4,4,2,1,3},
        { 3,3,1,2},
        { 2,2,1},
        { 1,1},
    };

    static const unsigned char codtab[TOTRUN_NUM][16] =
    {
        {1,3,2,3,2,3,2,3,2,3,2,3,2,3,2,1},
        {7,6,5,4,3,5,4,3,2,3,2,3,2,1,0},
        {5,7,6,5,4,3,4,3,2,3,2,1,1,0},
        {3,7,5,4,6,5,4,3,3,2,2,1,0},
        {5,4,3,7,6,5,4,3,2,1,1,0},
        {1,1,7,6,5,4,3,2,1,1,0},
        {1,1,5,4,3,3,2,1,1,0},
        {1,1,1,3,3,2,2,1,0},
        {1,0,1,3,2,1,1,1,},
        {1,0,1,3,2,1,1,},
        {0,1,1,2,1,3},
        {0,1,1,1,1},
        {0,1,1,1},
        {0,1,1},
        {0,1},
    };

    int code;
    int vlcnum = sym->value1;
    int retval = code_from_bitstream_2d(sym, currStream, &lentab[vlcnum][0], &codtab[vlcnum][0], 16, 1, &code);

    if (retval)
    {
        printf("ERROR: failed to find Total Zeros !cdc\n");
        exit(-1);
    }

    return retval;
}

/*!
 ************************************************************************
 * \brief
 *  Reads bits from the bitstream buffer
 *
 * \param buffer
 *    buffer containing VLC-coded data bits
 * \param totbitoffset
 *    bit offset from start of partition
 * \param bitcount
 *    total bytes in bitstream
 * \param numbits
 *    number of bits to read
 *
 ************************************************************************
 */

int ShowBits (unsigned char buffer[],int totbitoffset,int bitcount, int numbits){

    if ((totbitoffset + numbits )  > bitcount){
        return -1;
    }else{
        int bitoffset  = 7 - (totbitoffset & 0x07); // bit from start of unsigned char
        int byteoffset = (totbitoffset >> 3); // byte from start of buffer
        unsigned char *curbyte  = &(buffer[byteoffset]);
        int inf        = 0;

        while (numbits--)
        {
            inf <<=1;    
            inf |= ((*curbyte)>> (bitoffset--)) & 0x01;

            if (bitoffset == -1 ) 
            { //Move onto next byte to get all of numbits
                curbyte++;
                bitoffset = 7;
            }
        }
        return inf;           // return absolute offset in bit from start of frame
    }
}

/*!
 ************************************************************************
 * \brief
 *    read Level VLC codeword from UVLC-partition
 ************************************************************************
 */
int readSyntaxElement_Level_VLCN(SyntaxElement *sym, int vlc, Bitstream *currStream){
    int frame_bitoffset        = currStream->frame_bitoffset;
    int BitstreamLengthInBytes = currStream->bitstream_length;
    int BitstreamLengthInBits  = (BitstreamLengthInBytes << 3) + 7;
    unsigned char *buf                  = currStream->streamBuffer;

    int levabs, sign;
    int len = 1;
    int code = 1, sb;

    int shift = vlc - 1;

    // read pre zeros
    while (!ShowBits(buf, frame_bitoffset ++, BitstreamLengthInBits, 1))
        len++;

    frame_bitoffset -= len;

    if (len < 16)
    {
        levabs = ((len - 1) << shift) + 1;

        // read (vlc-1) bits -> suffix
        if (shift)
        {
            sb =  ShowBits(buf, frame_bitoffset + len, BitstreamLengthInBits, shift);
            code = (code << (shift) )| sb;
            levabs += sb;
            len += (shift);
        }

        // read 1 bit -> sign
        sign = ShowBits(buf, frame_bitoffset + len, BitstreamLengthInBits, 1);
        code = (code << 1)| sign;
        len ++;
    }
    else // escape
    {
        int addbit = len - 5;
        int offset = (1 << addbit) + (15 << shift) - 2047;

        sb = ShowBits(buf, frame_bitoffset + len, BitstreamLengthInBits, addbit);
        code = (code << addbit ) | sb;
        len   += addbit;

        levabs = sb + offset;

        // read 1 bit -> sign
        sign = ShowBits(buf, frame_bitoffset + len, BitstreamLengthInBits, 1);

        code = (code << 1)| sign;

        len++;
    }

    sym->inf = (sign)? -levabs : levabs;
    sym->len = len;

    currStream->frame_bitoffset = frame_bitoffset + len;


    return 0;
}

/*!
 ************************************************************************
 * \brief
 *    read Level VLC0 codeword from UVLC-partition
 ************************************************************************
 */
int readSyntaxElement_Level_VLC0(SyntaxElement *sym, Bitstream *currStream){
    int frame_bitoffset        = currStream->frame_bitoffset;
    int BitstreamLengthInBytes = currStream->bitstream_length;
    int BitstreamLengthInBits  = (BitstreamLengthInBytes << 3) + 7;
    unsigned char *buf                  = currStream->streamBuffer;
    int len = 1, sign = 0, level = 0, code = 1;

    while (!ShowBits(buf, frame_bitoffset++, BitstreamLengthInBits, 1))
        len++;

    if (len < 15)
    {
        sign  = (len - 1) & 1;
        level = ((len - 1) >> 1) + 1;
    }
    else if (len == 15)
    {
        // escape code
        code <<= 4;
        code |= ShowBits(buf, frame_bitoffset, BitstreamLengthInBits, 4);
        len  += 4;
        frame_bitoffset += 4;
        sign = (code & 0x01);
        level = ((code >> 1) & 0x07) + 8;
    }
    else if (len >= 16)
    {
        // escape code
        int addbit = (len - 16);
        int offset = (2048 << addbit) - 2032;
        len   -= 4;
        code   = ShowBits(buf, frame_bitoffset, BitstreamLengthInBits, len);
        sign   = (code & 0x01);
        frame_bitoffset += len;    
        level = (code >> 1) + offset;

        code |= (1 << (len)); // for display purpose only
        len += addbit + 16;
    }

    sym->inf = (sign) ? -level : level ;
    sym->len = len;

    currStream->frame_bitoffset = frame_bitoffset;
    return 0;
}

/*!
 ************************************************************************
 * \brief
 *  Reads bits from the bitstream buffer
 *
 * \param buffer
 *    containing VLC-coded data bits
 * \param totbitoffset
 *    bit offset from start of partition
 * \param info
 *    returns value of the read bits
 * \param bitcount
 *    total bytes in bitstream
 * \param numbits
 *    number of bits to read
 *
 ************************************************************************
 */
int GetBits (unsigned char buffer[],int totbitoffset,int *info, int bitcount, int numbits){

    if ((totbitoffset + numbits ) > bitcount){
        return -1;
    }else{
        int bitoffset  = 7 - (totbitoffset & 0x07); // bit from start of byte
        int byteoffset = (totbitoffset >> 3); // byte from start of buffer
        int bitcounter = numbits;
        unsigned char *curbyte  = &(buffer[byteoffset]);
        int inf = 0;

        while (numbits--)
        {
            inf <<=1;    
            inf |= ((*curbyte)>> (bitoffset--)) & 0x01;    
            if (bitoffset == -1 ) 
            { //Move onto next byte to get all of numbits
                curbyte++;
                bitoffset = 7;
            }
            // Above conditional could also be avoided using the following:
            // curbyte   -= (bitoffset >> 3);
            // bitoffset &= 0x07;
        }
        *info = inf;

        return bitcounter;           // return absolute offset in bit from start of frame
    }
}


/*!
 ************************************************************************
 * \brief
 *    read FLC codeword from UVLC-partition
 ************************************************************************
 */
int readSyntaxElement_FLC(SyntaxElement *sym, Bitstream *currStream){
    int BitstreamLengthInBits  = (currStream->bitstream_length << 3) + 7;

    if ((GetBits(currStream->streamBuffer, currStream->frame_bitoffset, &(sym->inf), BitstreamLengthInBits, sym->len)) < 0)
        return -1;

    sym->value1 = sym->inf;
    currStream->frame_bitoffset += sym->len; // move bitstream pointer


    return 1;
}

/*!
 ************************************************************************
 * \brief
 *    read NumCoeff/TrailingOnes codeword from UVLC-partition
 ************************************************************************
 */

int readSyntaxElement_NumCoeffTrailingOnes(SyntaxElement *sym, Bitstream *currStream){
    int frame_bitoffset        = currStream->frame_bitoffset;
    int BitstreamLengthInBytes = currStream->bitstream_length;
    int BitstreamLengthInBits  = (BitstreamLengthInBytes << 3) + 7;
    unsigned char *buf                  = currStream->streamBuffer;

    static const unsigned char lentab[3][4][17] =
    {
        {   // 0702
            { 1, 6, 8, 9,10,11,13,13,13,14,14,15,15,16,16,16,16},
            { 0, 2, 6, 8, 9,10,11,13,13,14,14,15,15,15,16,16,16},
            { 0, 0, 3, 7, 8, 9,10,11,13,13,14,14,15,15,16,16,16},
            { 0, 0, 0, 5, 6, 7, 8, 9,10,11,13,14,14,15,15,16,16},
        },
        {
            { 2, 6, 6, 7, 8, 8, 9,11,11,12,12,12,13,13,13,14,14},
            { 0, 2, 5, 6, 6, 7, 8, 9,11,11,12,12,13,13,14,14,14},
            { 0, 0, 3, 6, 6, 7, 8, 9,11,11,12,12,13,13,13,14,14},
            { 0, 0, 0, 4, 4, 5, 6, 6, 7, 9,11,11,12,13,13,13,14},
        },
        {
            { 4, 6, 6, 6, 7, 7, 7, 7, 8, 8, 9, 9, 9,10,10,10,10},
            { 0, 4, 5, 5, 5, 5, 6, 6, 7, 8, 8, 9, 9, 9,10,10,10},
            { 0, 0, 4, 5, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9,10,10,10},
            { 0, 0, 0, 4, 4, 4, 4, 4, 5, 6, 7, 8, 8, 9,10,10,10},
        },
    };

    static const unsigned char codtab[3][4][17] =
    {
        {
            { 1, 5, 7, 7, 7, 7,15,11, 8,15,11,15,11,15,11, 7,4},
            { 0, 1, 4, 6, 6, 6, 6,14,10,14,10,14,10, 1,14,10,6},
            { 0, 0, 1, 5, 5, 5, 5, 5,13, 9,13, 9,13, 9,13, 9,5},
            { 0, 0, 0, 3, 3, 4, 4, 4, 4, 4,12,12, 8,12, 8,12,8},
        },
        {
            { 3,11, 7, 7, 7, 4, 7,15,11,15,11, 8,15,11, 7, 9,7},
            { 0, 2, 7,10, 6, 6, 6, 6,14,10,14,10,14,10,11, 8,6},
            { 0, 0, 3, 9, 5, 5, 5, 5,13, 9,13, 9,13, 9, 6,10,5},
            { 0, 0, 0, 5, 4, 6, 8, 4, 4, 4,12, 8,12,12, 8, 1,4},
        },
        {
            {15,15,11, 8,15,11, 9, 8,15,11,15,11, 8,13, 9, 5,1},
            { 0,14,15,12,10, 8,14,10,14,14,10,14,10, 7,12, 8,4},
            { 0, 0,13,14,11, 9,13, 9,13,10,13, 9,13, 9,11, 7,3},
            { 0, 0, 0,12,11,10, 9, 8,13,12,12,12, 8,12,10, 6,2},
        },
    };

    int retval = 0, code;
    int vlcnum = sym->value1;
    // vlcnum is the index of Table used to code coeff_token
    // vlcnum==3 means (8<=nC) which uses 6bit FLC

    if (vlcnum == 3)
    {
        // read 6 bit FLC
        //code = ShowBits(buf, frame_bitoffset, BitstreamLengthInBytes, 6);
        code = ShowBits(buf, frame_bitoffset, BitstreamLengthInBits, 6);
        currStream->frame_bitoffset += 6;
        sym->value2 = (code & 3);
        sym->value1 = (code >> 2);

        if (!sym->value1 && sym->value2 == 3)
        {
            // #c = 0, #t1 = 3 =>  #c = 0
            sym->value2 = 0;
        }
        else
            sym->value1++;

        sym->len = 6;
    }
    else
    {
	
        //retval = code_from_bitstream_2d(sym, currStream, &lentab[vlcnum][0][0], &codtab[vlcnum][0][0], 17, 4, &code);    
        retval = code_from_bitstream_2d(sym, currStream, lentab[vlcnum][0], codtab[vlcnum][0], 17, 4, &code);
        if (retval)
        {
            printf("ERROR: failed to find NumCoeff/TrailingOnes\n");
            exit(-1);
        }
    }

    return retval;
}

/*!
 ************************************************************************
 * \brief
 *    get neighboring 4x4 block
 * \param pix
 *    returns position informations
 ************************************************************************
 */
void getNeighbourNNZ(int X,int Y,int refB,int xN, int yN, PixelPos *pix,unsigned char *BlockModeFlag){
    switch(refB){
        case 0:
            // left
            if(xN < 0){
                pix->available = 0;
            }else{
                if((BlockModeFlag[(Y>>1)*Block88Width + ((X-1)>>1)] == INTRA_B)){
                    pix->available = 1;
                }else{
                    pix->available = 0;
                }
            }
            break;
        case 1:
            // top
            if(yN < 0){
                pix->available = 0;
            }else{
                if((BlockModeFlag[((Y-1)>>1)*Block88Width + (X>>1)] == INTRA_B)){
                    pix->available = 1;
                }else{
                    pix->available = 0;
                }
            }
            break;
        default:
            pix->available = 0;
            break;
    }


}


/*!
 ************************************************************************
 * \brief
 *    Get the Prediction from the Neighboring Blocks for Number of 
 *    Nonzero Coefficients
 *
 *    Luma Blocks
 ************************************************************************
 */
int predict_nnz(int X,int Y, EncodedFrame *encodedFrame, unsigned char *BlockModeFlag){
    int pred_nnz = 0;
    int cnt = 0;
    PixelPos pix;
    int ioff = (X<<2);
    int joff = (Y<<2);
    BlockData  *refblock;
    int refblockpos;

    // left
    getNeighbourNNZ(X,Y,0,ioff-1,joff,&pix,BlockModeFlag);

    if(pix.available){
        refblockpos = Y*Block44Width + (X-1);
        //printf("ref left block = %d\n",refblockpos);
        refblock = &(encodedFrame->blockInfo[refblockpos]);
        pred_nnz += refblock->nonzero;
        cnt++;
    }

    // top
    getNeighbourNNZ(X,Y,1,ioff,joff-1,&pix,BlockModeFlag);
    if(pix.available){
        refblockpos = (Y-1)*Block44Width + X;
        //printf("ref top block = %d\n",refblockpos);
        refblock = &(encodedFrame->blockInfo[refblockpos]);
        pred_nnz += refblock->nonzero;
        cnt++;
    }

    if (cnt==2)
    {
        pred_nnz++;
        pred_nnz>>=1;
    }

    return pred_nnz;
}

/*!
 ************************************************************************
 * \brief
 *    Reads coeff of an 4x4 block (CAVLC)
 *
 * \author
 *    Karl Lillevold <karll@real.com>
 *    contributions by James Au <james@ubvideo.com>
 ************************************************************************
 */
void readCoeff4x4_CAVLC (int X, int Y, int levarr[BLOCKSIZE44], int runarr[BLOCKSIZE44], int *number_coefficients, EncodedFrame *encodedFrame){
    SyntaxElement currSE;
    static Bitstream *currStream = &IntraBS;
    int block44pos               = Y * Block44Width + X;
    BlockData *currblock         = &(encodedFrame->blockInfo[block44pos]);

    int k, code, vlcnum;
    static int numcoeff, numtrailingones, numcoeff_vlc;
    static int level_two_or_higher;
    int numones, totzeros=0, abslevel;
    int zerosleft, ntr, dptype = 0;
    int max_coeff_num = 0, nnz;
    static int incVlc[] = {0,3,6,12,24,48,32768};    // maximum vlc = 6

    numcoeff = 0;

    max_coeff_num = BLOCKSIZE44;
    dptype = SE_LUM_AC_INTRA;
    currblock->nonzero = 0; 

    currSE.type = dptype;

    nnz = predict_nnz(X,Y,encodedFrame,encodedFrame->BlockModeFlag);
	//printf("nnz = %d\n",nnz);

    if (nnz < 2)
    {
        numcoeff_vlc = 0;
    }
    else if (nnz < 4)
    {
        numcoeff_vlc = 1;
    }
    else if (nnz < 8)
    {
        numcoeff_vlc = 2;
    }
    else
    {
        numcoeff_vlc = 3;
    }
	//printf("table %d\n",numcoeff_vlc);
    currSE.value1 = numcoeff_vlc;
	
    readSyntaxElement_NumCoeffTrailingOnes(&currSE, currStream);

    numcoeff        =  currSE.value1;
    numtrailingones =  currSE.value2;
    

    currblock->nonzero = numcoeff;
	//printf("nonzero = %d\n",currblock->nonzero);

    memset(levarr, 0, max_coeff_num * sizeof(int));
    memset(runarr, 0, max_coeff_num * sizeof(int));

    numones = numtrailingones;
    *number_coefficients = numcoeff;

    if (numcoeff)
    {
        if (numtrailingones)
        {      
            currSE.len = numtrailingones;


            readSyntaxElement_FLC (&currSE, currStream);

            code = currSE.inf;
            ntr = numtrailingones;
            for (k = numcoeff - 1; k > numcoeff - 1 - numtrailingones; k--)
            {
                ntr --;
                levarr[k] = (code>>ntr)&1 ? -1 : 1;
            }
        }

        // decode levels
        level_two_or_higher = (numcoeff > 3 && numtrailingones == 3)? 0 : 1;
        vlcnum = (numcoeff > 10 && numtrailingones < 3) ? 1 : 0;

        for (k = numcoeff - 1 - numtrailingones; k >= 0; k--)
        {


            if (vlcnum == 0)
                readSyntaxElement_Level_VLC0(&currSE, currStream);
            else
                readSyntaxElement_Level_VLCN(&currSE, vlcnum, currStream);

            if (level_two_or_higher)
            {
                currSE.inf += (currSE.inf > 0) ? 1 : -1;
                level_two_or_higher = 0;
            }

            levarr[k] = currSE.inf;
            abslevel = iabs(levarr[k]);
            if (abslevel  == 1)
                numones ++;

            // update VLC table
            if (abslevel  > incVlc[vlcnum])
                vlcnum++;

            if (k == numcoeff - 1 - numtrailingones && abslevel >3)
                vlcnum = 2;      
        }

        if (numcoeff < max_coeff_num)
        {
            // decode total run
            vlcnum = numcoeff - 1;
            currSE.value1 = vlcnum;

            readSyntaxElement_TotalZeros(&currSE, currStream);

            totzeros = currSE.value1;
        }
        else
        {
            totzeros = 0;
        }

        // decode run before each coefficient
        zerosleft = totzeros;
        int temp_i = numcoeff - 1;

        if (zerosleft > 0 && temp_i > 0)
        {
            do
            {
                // select VLC for runbefore
                vlcnum = imin(zerosleft - 1, RUNBEFORE_NUM_M1);

                currSE.value1 = vlcnum;

                readSyntaxElement_Run(&currSE, currStream);
                runarr[temp_i] = currSE.value1;

                zerosleft -= runarr[temp_i];
                temp_i --;
            } while (zerosleft != 0 && temp_i != 0);
        }
        runarr[temp_i] = zerosleft;    
    } // if numcoeff
    
    //printf("totzeros = %d, numcoeff = %d, numtrailingones = %d\n",totzeros,numcoeff,numtrailingones);
}

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "vlc.h"
#include "inlineFunc.h"


#define SE_LUM_AC_INTRA 9
#define SE_HEADER 0

//AVC Profile IDC definitions
#define BASELINE         66      //!< YUV 4:2:0/8  "Baseline"
#define MAIN             77      //!< YUV 4:2:0/8  "Main"
#define EXTENDED         88      //!< YUV 4:2:0/8  "Extended"
#define FREXT_HP        100      //!< YUV 4:2:0/8 "High"
#define FREXT_Hi10P     110      //!< YUV 4:2:0/10 "High 10"
#define FREXT_Hi422     122      //!< YUV 4:2:2/10 "High 4:2:2"
#define FREXT_Hi444     244      //!< YUV 4:4:4/14 "High 4:4:4"
#define FREXT_CAVLC444   44      //!< YUV 4:4:4/14 "CAVLC 4:4:4"


#define IS_FREXT_PROFILE(profile_idc) ( profile_idc>=FREXT_HP || profile_idc == FREXT_CAVLC444 )

#define LEVEL_NUM         6
#define TOTRUN_NUM       15
#define RUNBEFORE_NUM     7
#define RUNBEFORE_NUM_M1  6


//! Syntax Element
typedef struct syntaxelement{
    int                 type;           //!< type of syntax element for data part.
    int                 value1;         //!< numerical value of syntax element
    int                 value2;         //!< for blocked symbols, e.g. run/level
    int                 len;            //!< length of code
    int                 inf;            //!< info part of UVLC code
    unsigned int        bitpattern;     //!< UVLC bitpattern
    int                 context;        //!< CABAC context
} SyntaxElement;


/*!
 ************************************************************************
 * \brief
 *    Makes code word and passes it back
 *
 * \par Input:
 *    Info   : Xn..X2 X1 X0                                             \n
 *    Length : Total number of bits in the codeword
 ************************************************************************
 */

int symbol2vlc(SyntaxElement *sym)
{
    int info_len = sym->len;

    // Convert info into a bitpattern int
    sym->bitpattern = 0;

    // vlc coding
    while(--info_len >= 0)
    {
        sym->bitpattern <<= 1;
        sym->bitpattern |= (0x01 & (sym->inf >> info_len));
    }
 
    //printf("pattern = ");
    //printbinary(sym->bitpattern,sym->len);


    return 0;
}




/*!
 ************************************************************************
 * \brief
 *    writes UVLC code to the appropriate buffer
 ************************************************************************
 */
void  writeUVLC2buffer(SyntaxElement *se, Bitstream *currStream)
{
    unsigned int mask = 1 << (se->len - 1);
    unsigned char *byte_buf  = &(currStream->byte_buf);
    int *bits_to_go = &currStream->bits_to_go;
    int i;
   
    //printf("pattern = ");
    //printbinary(se->bitpattern,se->len);
   
   
    // Add the new bits to the bitstream.
    // Write out a byte if it is full
    if ( se->len < 33 )
    {
        for (i = 0; i < se->len; i++)
        {
            *byte_buf <<= 1;

            if (se->bitpattern & mask)
                *byte_buf |= 1;

            mask >>= 1;

            if ((--(*bits_to_go)) == 0)
            {
                *bits_to_go = 8;      
                currStream->streamBuffer[currStream->byte_pos++] = *byte_buf;
               // printf("write byte to buffer: ");
                //printbinary((int)(*byte_buf),8);
                *byte_buf = 0;
            }
        }
    }
    else
    {
        // zeros
        for (i = 0; i < (se->len - 32); i++)
        {
            *byte_buf <<= 1;

            if ((--(*bits_to_go)) == 0)
            {
                *bits_to_go = 8;      
                currStream->streamBuffer[currStream->byte_pos++] = *byte_buf;
                //printf("write byte to buffer: ");
                //printbinary((int)(*byte_buf),8);
                *byte_buf = 0;      
            }
        }
        // actual info
        mask = 1 << 31;
        for (i = 0; i < 32; i++)
        {
            *byte_buf <<= 1;

            if (se->bitpattern & mask)
                *byte_buf |= 1;

            mask >>= 1;

            if ((--(*bits_to_go)) == 0)
            {
                *bits_to_go = 8;      
                currStream->streamBuffer[currStream->byte_pos++] = *byte_buf;
                //printf("write byte to buffer: ");
                //printbinary((int)(*byte_buf),8);
                *byte_buf = 0;      
            }
        }
    }
   /* 
    if((*bits_to_go)!=8){
        printf("buffer bits: ");
        printbinary((int)(*byte_buf),(8-*bits_to_go));
    }*/

}


/*!
 ************************************************************************
 * \brief
 *    write VLC for NumCoeff and TrailingOnes
 ************************************************************************
 */

int writeSyntaxElement_NumCoeffTrailingOnes(SyntaxElement *se, Bitstream *currStream,int write)
{
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
    int vlcnum = se->len;

    // se->value1 : numcoeff
    // se->value2 : numtrailingones

    if (vlcnum == 3)
    {
        se->len = 6;  // 4 + 2 bit FLC
        if (se->value1 > 0)
        {
            se->inf = ((se->value1-1) << 2) | se->value2;
        }
        else
        {
            se->inf = 3;
        }
    }
    else
    {
        se->len = lentab[vlcnum][se->value2][se->value1];
        se->inf = codtab[vlcnum][se->value2][se->value1];
    }

    if (se->len == 0)
    {
        printf("ERROR: (numcoeff,trailingones) not valid: vlc=%d (%d, %d)\n",
                vlcnum, se->value1, se->value2);
        exit(-1);
    }

    symbol2vlc(se);
    if(write){
		//printf("len = %d, code = %d\n",se->len,se->inf);
        writeUVLC2buffer(se, currStream);
    }
    /*
    if(se->type != SE_HEADER){
        currStream->write_flag = 1;
    }*/
    return (se->len);
}

/*!
 ************************************************************************
 * \brief
 *    generates VLC code and passes the codeword to the buffer
 ************************************************************************
 */
int writeSyntaxElement_VLC(SyntaxElement *se, Bitstream *currStream, int write)
{
    se->inf = se->value1;
    se->len = se->value2;
    symbol2vlc(se);
    
    if(write){
        writeUVLC2buffer(se, currStream);
    }
    /*
    if(se->type != SE_HEADER)
        currStream->write_flag = 1;
    */
    return (se->len);
}




/*!
 ************************************************************************
 * \brief
 *    write VLC for Coeff Level (VLC1)
 ************************************************************************
 */
int writeSyntaxElement_Level_VLC1(SyntaxElement *se, Bitstream *currStream, int profile_idc,int write)
{
    int level  = se->value1;
    int sign   = (level < 0 ? 1 : 0);
    int levabs = iabs(level);

    if (levabs < 8)
    {
        se->len = levabs * 2 + sign - 1;
        se->inf = 1;
    }
    else if (levabs < 16) 
    {
        // escape code1
        se->len = 19;
        se->inf = 16 | ((levabs << 1) - 16) | sign;
    }
    else
    {
        int iMask = 4096, numPrefix = 0;
        int levabsm16 = levabs + 2032;

        // escape code2
        if ((levabsm16) >= 4096)
        {
            numPrefix++;
            while ((levabsm16) >= (4096 << numPrefix))
            {
                numPrefix++;
            }
        }

        iMask <<= numPrefix;
        se->inf = iMask | ((levabsm16 << 1) - iMask) | sign;

        /* Assert to make sure that the code fits in the VLC */
        /* make sure that we are in High Profile to represent level_prefix > 15 */
        if (numPrefix > 0 && !IS_FREXT_PROFILE( profile_idc ))
        {
            //error( "level_prefix must be <= 15 except in High Profile\n",  1000 );
            se->len = 0x0000FFFF; // This can be some other big number
            return (se->len);
        }

        se->len = 28 + (numPrefix << 1);
    }

    symbol2vlc(se);
    
    if(write){
        writeUVLC2buffer(se, currStream);
    }
    /*
    if(se->type != SE_HEADER)
        currStream->write_flag = 1;
    */
    return (se->len);
}



/*!
 ************************************************************************
 * \brief
 *    write VLC for Coeff Level
 ************************************************************************
 */
int writeSyntaxElement_Level_VLCN(SyntaxElement *se, int vlc, Bitstream *currStream, int profile_idc,int write)
{  
    int level  = se->value1;
    int sign   = (level < 0 ? 1 : 0);
    int levabs = iabs(level) - 1;  

    int shift = vlc - 1;        
    int escape = (15 << shift);

    if (levabs < escape)
    {
        int sufmask   = ~((0xffffffff) << shift);
        int suffix    = (levabs) & sufmask;

        se->len = ((levabs) >> shift) + 1 + vlc;
        se->inf = (2 << shift) | (suffix << 1) | sign;
    }
    else
    {
        int iMask = 4096;
        int levabsesc = levabs - escape + 2048;
        int numPrefix = 0;

        if ((levabsesc) >= 4096)
        {
            numPrefix++;
            while ((levabsesc) >= (4096 << numPrefix))
            {
                numPrefix++;
            }
        }

        iMask <<= numPrefix;
        se->inf = iMask | ((levabsesc << 1) - iMask) | sign;

        /* Assert to make sure that the code fits in the VLC */
        /* make sure that we are in High Profile to represent level_prefix > 15 */
        if (numPrefix > 0 &&  !IS_FREXT_PROFILE( profile_idc ))
        {
            se->len = 0x0000FFFF; // This can be some other big number
            return (se->len);
        }
        se->len = 28 + (numPrefix << 1);
    }

    symbol2vlc(se);
    
    if(write){
        writeUVLC2buffer(se, currStream);
    }   
    /*
    if(se->type != SE_HEADER)
        currStream->write_flag = 1;
    */
    return (se->len);
}



/*!
 ************************************************************************
 * \brief
 *    write VLC for TotalZeros
 ************************************************************************
 */
int writeSyntaxElement_TotalZeros(SyntaxElement *se, Bitstream *currStream,int write)
{
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
    int vlcnum = se->len;

    //se->value1 : TotalZeros
    se->len = lentab[vlcnum][se->value1];
    se->inf = codtab[vlcnum][se->value1];

    if (se->len == 0)
    {
        printf("ERROR: (TotalZeros) not valid: (%d)\n",se->value1);
        exit(-1);
    }

    symbol2vlc(se);
    
    if(write){
        writeUVLC2buffer(se, currStream);
    }
    /*
    if(se->type != SE_HEADER)
        currStream->write_flag = 1;
    */
    return (se->len);
}



/*!
 ************************************************************************
 * \brief
 *    write VLC for Run Before Next Coefficient, VLC0
 ************************************************************************
 */
int writeSyntaxElement_Run(SyntaxElement *se, Bitstream *currStream,int write)
{
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
    int vlcnum = se->len;

    // se->value1 : run
    se->len = lentab[vlcnum][se->value1];
    se->inf = codtab[vlcnum][se->value1];

    if (se->len == 0)
    {
        printf("ERROR: (run) not valid: (%d)\n",se->value1);
        exit(-1);
    }

    symbol2vlc(se);
    
    if(write){
        writeUVLC2buffer(se, currStream);
    }
    /*
    if(se->type != SE_HEADER)
        currStream->write_flag = 1;
    */

    return (se->len);
}

void getNeighbourNNZ(int X,int Y,int refB,int xN, int yN, PixelPos *pix, unsigned char *intraBlockFlag){
    switch(refB){
        case 0:
            // left
            if(xN < 0){
                pix->available = 0;
            }else{
                if(intraBlockFlag[(Y>>1)*Block88Width + ((X-1)>>1)] == 1){
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
				if(intraBlockFlag[((Y-1)>>1)*Block88Width + (X>>1)] == 1){
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

int predict_nnz(int X,int Y, EncodedFrame *encodedFrame){
    int pred_nnz = 0;
    int cnt = 0;
    PixelPos pix;
    int ioff = (X<<2);
    int joff = (Y<<2);
    BlockData  *refblock;
    int refblockpos;

    // left
    getNeighbourNNZ(X,Y,0,ioff-1,joff,&pix,encodedFrame->intraBlockFlag);    
    
    if(pix.available){
        refblockpos = Y*Block44Width + (X-1);
        //printf("ref left block = %d\n",refblockpos);
        refblock = &(encodedFrame->blockInfo[refblockpos]);
        pred_nnz += refblock->nonzero;
        cnt++;
    }

    // top 
    getNeighbourNNZ(X,Y,1,ioff,joff-1,&pix,encodedFrame->intraBlockFlag);    
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
 *    Writes coeff of an 4x4 block (CAVLC)
 *
 * \author (from JM 15.1)
 *    Karl Lillevold <karll@real.com>
 *    contributions by James Au <james@ubvideo.com>
 ************************************************************************
 */
int writeCoeff4x4_CAVLC(EncodedFrame *encodedFrame, int X,int Y, int block44pos, int write){
    int           no_bits    = 0;
    SyntaxElement se;
    BlockData     *currblock = &(encodedFrame->blockInfo[block44pos]);
    int           *bitCount  = &(currblock->bitcounter);
    Bitstream *dataPart = &IntraBuffer;

    int k,level = 1,run,vlcnum;
    int numcoeff = 0, lastcoeff = 0, numtrailingones = 0; 
    int numones = 0, totzeros = 0, zerosleft, numcoef;
    int numcoeff_vlc;
    int code, level_two_or_higher;
    int dptype = 0;
    int nnz, max_coeff_num = 0;

    static const int incVlc[] = {0, 3, 6, 12, 24, 48, 32768};  // maximum vlc = 6


    int*  pLevel = NULL;
    int*  pRun = NULL;
    


    max_coeff_num = 16;
    //bitcounttype = BITS_COEFF_Y_MB;
    (*bitCount) = 0;

    pLevel = currblock->level;
    pRun   = currblock->run;
    dptype = SE_LUM_AC_INTRA;
	

    for(k = 0; (k <= 16) && level != 0; k++)
    {
        level = pLevel[k]; // level
        run   = pRun[k];   // run

        if (level)
        {

            totzeros += run; // lets add run always (even if zero) to avoid conditional
            if (iabs(level) == 1)
            {
                numones ++;
                numtrailingones ++;
                numtrailingones = imin(numtrailingones, 3); // clip to 3
            }
            else
            {
                numtrailingones = 0;
            }
            numcoeff ++;
            lastcoeff = k;
        }
    }
    currblock->nonzero = numcoeff;
	
    
    nnz = predict_nnz(X,Y,encodedFrame);
    
    if (nnz < 2)	// choose vlc table, context-adaptive
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

    // total coeff coding
    se.type  = dptype;
    se.value1 = numcoeff;
    se.value2 = numtrailingones;
    se.len    = numcoeff_vlc; /* use len to pass vlcnum */
 
    writeSyntaxElement_NumCoeffTrailingOnes(&se, dataPart, write);
	

    (*bitCount) += se.len;
    no_bits     += se.len;
    if (!numcoeff)
        return no_bits;

    // total coeff ok

    if (numcoeff)
    {
        code = 0;
        for (k = lastcoeff; k > lastcoeff - numtrailingones; k--)
        {
            level = pLevel[k]; // level

            if (iabs(level) > 1)
            {
                printf("ERROR: level > 1\n");
                exit(-1);
            }

            code <<= 1;

            if (level < 0)
            {
                code |= 0x1;
            }
        } // TrailingOne sign coding OK, store in "code"

        if (numtrailingones)
        {
            se.type  = dptype;

            se.value2 = numtrailingones;
            se.value1 = code;

            //if(write)
            //    printf("T1s\t\t\t\t=>\t");
            
            writeSyntaxElement_VLC (&se, dataPart,write);

            (*bitCount) += se.len;
            no_bits     += se.len;

        }

        // encode levels
        level_two_or_higher = (numcoeff > 3 && numtrailingones == 3) ? 0 : 1;

        vlcnum = (numcoeff > 10 && numtrailingones < 3) ? 1 : 0;

        for (k = lastcoeff - numtrailingones; k >= 0; k--)
        {
            level = pLevel[k]; // level

            se.value1 = level;
            se.type  = dptype;


            if (level_two_or_higher)	// save bitrate, Note 1 in example
            {
                level_two_or_higher = 0;

                if (se.value1 > 0)
                    se.value1 --;
                else
                    se.value1 ++;        
            }


            //    encode level
            //if(write)
            //    printf("Level(%3d)\t\t=>\t",level);
            if (vlcnum == 0)
                writeSyntaxElement_Level_VLC1(&se, dataPart, MAIN, write);	// suffixlength initial to 0
            else
                writeSyntaxElement_Level_VLCN(&se, vlcnum, dataPart, MAIN, write); // suffixlength initial to 1


            // update VLC table
            if (iabs(level) > incVlc[vlcnum])
                vlcnum++;

            if ((k == lastcoeff - numtrailingones) && iabs(level) > 3)
                vlcnum = 2;

            (*bitCount)  += se.len;
            no_bits      += se.len;
        }

        // encode total zeroes
        if (numcoeff < max_coeff_num)
        {

            se.type  = dptype;
            se.value1 = totzeros;

            vlcnum = numcoeff - 1;

            se.len = vlcnum;
            //if(write)
            //    printf("Total zeros(%d)\t=>\t",totzeros);
            writeSyntaxElement_TotalZeros(&se, dataPart, write);

            (*bitCount) += se.len;
            no_bits     += se.len;
        }

        // encode run before each coefficient
        zerosleft = totzeros;
        numcoef = numcoeff;
        for (k = lastcoeff; k >= 0; k--)
        {
            run = pRun[k]; // run

            se.value1 = run;
            se.type   = dptype;

            // for last coeff, run is remaining totzeros
            // when zerosleft is zero, remaining coeffs have 0 run
            if ((!zerosleft) || (numcoeff <= 1 ))
                break;

            if (numcoef > 1 && zerosleft)
            {
                vlcnum = imin(zerosleft - 1, RUNBEFORE_NUM_M1);
                se.len = vlcnum;

                //if(write)
                //    printf("Run before(%d)\t=>\t",run);
                writeSyntaxElement_Run(&se, dataPart,write);

                (*bitCount) += se.len;
                no_bits     += se.len;

                zerosleft -= run;
                numcoef --;
            }
        }
    }

    return no_bits;
}

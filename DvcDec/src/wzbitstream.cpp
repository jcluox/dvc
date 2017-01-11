#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "global.h"
#include "error.h"
#include "reconstruction.h"
#include "LdpcaDecoder.h"
#include "jbig.h"
#include "sJBIG.h"

void readGOP(int begin, int end, FILE* fp){
	int frameIdx = (begin + end) / 2;
	if(frameIdx > begin){
		int SyndromeByteLength = Block44Num / 8;	//fixed length source for LDPCA

		//read AC range (2 bytes for each AC)
		unsigned char ACRangeU, ACRangeB;
		for(int j=1; j<BLOCKSIZE44; j++){
			int index = zigzag[j];
			if(Qtable[index] > 0){
				fread(&ACRangeU, sizeof(unsigned char), 1, fp);
				fread(&ACRangeB, sizeof(unsigned char), 1, fp);
				encodedFrames[frameIdx].ACRange[index] = ((int)ACRangeU)<<8;
				encodedFrames[frameIdx].ACRange[index] |= (int)ACRangeB;
			}
			else
				break;
		}

		//read block mode selection index
		encodedFrames[frameIdx].BlockModeFlag = (unsigned char*)malloc(sizeof(unsigned char) * Block88Num);
		memset(encodedFrames[frameIdx].BlockModeFlag, 0, sizeof(unsigned char) * Block88Num);
		//read skip block index
		fread(&(encodedFrames[frameIdx].skipMode), sizeof(unsigned char), 1, fp);
		if(encodedFrames[frameIdx].skipMode == 1){
			/*int numBytes = (int)ceil((double)Block88Num / 8.0);
			unsigned char *skipBlockBytes = (unsigned char*)malloc(sizeof(unsigned char) * numBytes);
			fread(skipBlockBytes, sizeof(unsigned char), numBytes, fp);
			encodedFrames[frameIdx].skipBlockFlag = ByteToBit(skipBlockBytes, numBytes, Block88Num);*/
			// JBIG decoder
            int width = Block88Width;
            int height = Block88Height;
            int buffer_size = ( ((((Block88Width+4)>>3)<<3) * Block88Height) >> 3 ) * 2;
            int stripe = height;
			struct jbg_enc_state se;
            jbg_enc_init(&se, width, height, 1, NULL, NULL, NULL);
            jbg_enc_options(&se, JBG_ILEAVE | JBG_SMID, JBG_TPDON | JBG_DPON , stripe, 8, 0);
			unsigned char jbgh[20];
            jbg_set_header(&se, jbgh);

            unsigned char *jbigDec_data = (unsigned char *)malloc(sizeof(unsigned char) * buffer_size);
            unsigned char *jbigEnc_data = (unsigned char *)malloc(sizeof(unsigned char) * buffer_size);
            int read_byte = (int)fread(jbigEnc_data, sizeof(unsigned char), buffer_size, fp);
            int decode_byte = simple_JBIG_dec(jbigEnc_data, buffer_size, jbgh, buffer_size, jbigDec_data);
            encodedFrames[frameIdx].skipBlockBits = (decode_byte << 3) + 1;	//1 bit to represent the mode is enabled in the frame or not
            fseek(fp, -(read_byte-decode_byte), SEEK_CUR);
            unsigned char *skipBlockFlag = ByteToBitmap(jbigDec_data, width, height);
			for(int i=0; i<Block88Num; i++){
				if(skipBlockFlag[i] == 1)
					encodedFrames[frameIdx].BlockModeFlag[i] = SKIP_B;
			}
            free(jbigDec_data);
            free(jbigEnc_data);
		}
		else{
			encodedFrames[frameIdx].skipBlockBits = 1;	//1 bit to represent the mode is enabled in the frame or not
		}
		
	
		

		//read intra block index
		fread(&(encodedFrames[frameIdx].intraMode), sizeof(unsigned char), 1, fp);
		if(encodedFrames[frameIdx].intraMode == 1){
			// JBIG decoder
            int width = Block88Width;
            int height = Block88Height;
            int buffer_size = ( ((((Block88Width+4)>>3)<<3) * Block88Height) >> 3 ) * 2;
            int stripe = height;
			struct jbg_enc_state se;
            jbg_enc_init(&se, width, height, 1, NULL, NULL, NULL);
            jbg_enc_options(&se, JBG_ILEAVE | JBG_SMID, JBG_TPDON | JBG_DPON , stripe, 8, 0);
			unsigned char jbgh[20];
            jbg_set_header(&se, jbgh);

            unsigned char *jbigDec_data = (unsigned char *)malloc(sizeof(unsigned char) * buffer_size);
            unsigned char *jbigEnc_data = (unsigned char *)malloc(sizeof(unsigned char) * buffer_size);
            int read_byte = (int)fread(jbigEnc_data, sizeof(unsigned char), buffer_size, fp);
            int decode_byte = simple_JBIG_dec(jbigEnc_data, buffer_size, jbgh, buffer_size, jbigDec_data);
            encodedFrames[frameIdx].intraBlockBits = (decode_byte << 3) + 1;	//1 bit to represent the mode is enabled in the frame or not
            fseek(fp, -(read_byte-decode_byte), SEEK_CUR);
            unsigned char *intraBlockFlag = ByteToBitmap(jbigDec_data, width, height);
			for(int i=0; i<Block88Num; i++){
				if(intraBlockFlag[i] == 1)
					encodedFrames[frameIdx].BlockModeFlag[i] = INTRA_B;
			}
            free(jbigDec_data);
            free(jbigEnc_data);
		}
		else{
			encodedFrames[frameIdx].intraBlockBits = 1;	//1 bit to represent the mode is enabled in the frame or not
		}
		

		//read crc + syndrome of each bitplane
		encodedFrames[frameIdx].CRC = (unsigned char*)malloc(sizeof(unsigned char) * QbitsNumPerBlk);
		encodedFrames[frameIdx].AccumulatedSyndrome = (unsigned char**)malloc(sizeof(unsigned char*) * QbitsNumPerBlk);
		for(int j=0; j<QbitsNumPerBlk; j++){
			encodedFrames[frameIdx].AccumulatedSyndrome[j] = (unsigned char*)malloc(sizeof(unsigned char) * SyndromeByteLength);
			fread(&(encodedFrames[frameIdx].CRC[j]), sizeof(unsigned char), 1, fp);
			fread(encodedFrames[frameIdx].AccumulatedSyndrome[j], sizeof(unsigned char), SyndromeByteLength, fp);
		}

		if(encodedFrames[frameIdx].intraMode){
			//for intra block
			encodedFrames[frameIdx].blockInfo = (BlockData*)malloc(sizeof(BlockData) * Block44Num);
			memset(encodedFrames[frameIdx].blockInfo, 0, sizeof(BlockData) * Block44Num);
			codingOption.IntraModeExist = 1;
		}

		readGOP(begin, frameIdx, fp);
		readGOP(frameIdx, end, fp);
	}
}


void readWZBitstream(char *fileName){
	FILE *fp = fopen(fileName, "rb");
	if(fp == NULL){
		sprintf(errorText, "File %s doesn't exist", fileName);
		errorMsg(errorText);
	}
	printf("WZ bitstream file name: %s\n", fileName);

	unsigned char buf;
	//read frame resolution (1 byte)
	fread(&buf, sizeof(unsigned char), 1, fp);
	if(buf == 0){
		//QCIF
		codingOption.FrameWidth = 176;
		codingOption.FrameHeight = 144;
	}
	else if(buf == 1){
		//CIF
		codingOption.FrameWidth = 352;
		codingOption.FrameHeight = 288;
	}
	else{
		sprintf(errorText, "In parsing file %s: Unsupport resolution (support QCIF, CIF)", fileName);
		errorMsg(errorText);
	}
	FrameSize = codingOption.FrameWidth * codingOption.FrameHeight;
	Block44Num = FrameSize / BLOCKSIZE44;
	Block88Num = Block44Num / 4;
	Block1616Num = Block88Num / 4;
	Block44Height = codingOption.FrameHeight / 4;
	Block44Width = codingOption.FrameWidth / 4;
	Block88Height = codingOption.FrameHeight / 8;
	Block88Width = codingOption.FrameWidth / 8;
	Block1616Height = codingOption.FrameHeight / 16;
	Block1616Width = codingOption.FrameWidth / 16;

	//read Qindex (1 byte)
	fread(&buf, sizeof(unsigned char), 1, fp);
	codingOption.Qindex = buf;
	Qtable = QuantizationTable[codingOption.Qindex];
	QtableBits = QuantizationTableBits[codingOption.Qindex];
	for(int j=0; j<BLOCKSIZE44; j++)
		QbitsNumPerBlk += QtableBits[j];
	for(int j=1; j<BLOCKSIZE44; j++){
		if(Qtable[j] > 0)
			ACLevelShift[j] = (Qtable[j] - 2) / 2;
	}
	//sprintf(codingOption.OutputDecodedYuvFile, "%s0%d.yuv", codingOption.OutputDecodedYuvFile, codingOption.Qindex); //output sequence name
	//sprintf(codingOption.RDFile, "%s0%d.txt", codingOption.RDFile, codingOption.Qindex); //output rate and PSNR file
	sprintf(codingOption.OutputDecodedYuvFile, "%s.yuv", codingOption.OutputDecodedYuvFile); //output sequence name
	sprintf(codingOption.RDFile, "%s.txt", codingOption.RDFile); //output rate and PSNR file

	//read QP (1 byte)
    fread(&buf, sizeof(unsigned char), 1, fp);
    codingOption.IntraQP = buf;

	//read GOP size (1 byte)
	fread(&buf, sizeof(unsigned char), 1, fp);
	codingOption.GOP = buf;

	//read frame number (2 bytes)
	fread(&buf, sizeof(unsigned char), 1, fp);
	codingOption.FramesToBeDecoded = ((int)buf) << 8;
	fread(&buf, sizeof(unsigned char), 1, fp);
	codingOption.FramesToBeDecoded |= (int)buf;

	//read deblocking filter mode (1 byte)
	fread(&buf, sizeof(unsigned char), 1, fp);
	codingOption.DeblockingMode = buf;

	//read encoded WZ bitstream
	codingOption.IntraModeExist = 0;
	encodedFrames = (EncodedFrame*)malloc(sizeof(EncodedFrame) * codingOption.FramesToBeDecoded);
	for(int i=0; i<codingOption.FramesToBeDecoded; i+=codingOption.GOP){
		//for each GOP
		int begin = i;
		int end = i + codingOption.GOP;
		if(end >= codingOption.FramesToBeDecoded)
			end = codingOption.FramesToBeDecoded - 1;
		readGOP(begin, end, fp);
	}
	fclose(fp);
}
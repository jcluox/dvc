#include <stdio.h>
#include "global.h"
#include "quantization.h"
#include "LdpcaEncoder.h"
#include "sJBIG.h"

void outputGOP(int begin, int end, FILE* fp){
	int frameIdx = (begin + end) / 2;
	if(frameIdx > begin){
		//output AC range (2 bytes for each AC)
		unsigned char ACRangeU, ACRangeB;
		for(int j=1; j<BLOCKSIZE44; j++){
			int index = zigzag[j];
			if(Qtable[index] > 0){
				ACRangeU = (unsigned char)((encodedFrames[frameIdx].ACRange[index])>>8);
				ACRangeB = (unsigned char)(encodedFrames[frameIdx].ACRange[index]);
				fwrite(&ACRangeU, sizeof(unsigned char), 1, fp);
				fwrite(&ACRangeB, sizeof(unsigned char), 1, fp);
			}
			else
				break;
		}

		//output skip block binary decision map
		fwrite(&(encodedFrames[frameIdx].skipMode), sizeof(unsigned char), 1, fp);
		if(encodedFrames[frameIdx].skipMode == 1){
			/*int numBytes;
			unsigned char *skipBlockBytes = BitToByte(encodedFrames[frameIdx].skipBlockFlag, Block88Num, &numBytes);
			fwrite(skipBlockBytes, sizeof(unsigned char), numBytes, fp);*/
            // JBIG compression per frame
            /*int size = 0;
            size = ((Block88Num)+4) >> 3;
			printf("orignal size = %d bytes\n",size);
			ori_skip_byte += size;*/
			int bitmap_byte = 0;
            unsigned char *jbigdata = BitToBitmap(encodedFrames[frameIdx].skipBlockFlag, Block88Width, Block88Height, &bitmap_byte);
            unsigned char jbgh[20];
            int enc_one_bytes = simple_JBIG_enc(fp, jbigdata, Block88Width, Block88Height, jbgh, Block88Height, 1);
            //printf("encoding bytes = %d bytes\n",enc_one_bytes);
            //jbig_skip_byte += enc_one_bytes;
        }
		//output intra block binary decision map
		fwrite(&(encodedFrames[frameIdx].intraMode), sizeof(unsigned char), 1, fp);
		if(encodedFrames[frameIdx].intraMode == 1){
            // JBIG compression per frame
			int bitmap_byte = 0;
            unsigned char *jbigdata = BitToBitmap(encodedFrames[frameIdx].intraBlockFlag, Block88Width, Block88Height, &bitmap_byte);
            unsigned char jbgh[20];
            int enc_one_bytes = simple_JBIG_enc(fp, jbigdata, Block88Width, Block88Height, jbgh, Block88Height, 1);
        }


		//syndrome of each bitplane
		for(int j=0; j<QbitsNumPerBlk; j++){
			fwrite(&(encodedFrames[frameIdx].CRC[j]), sizeof(unsigned char), 1, fp);
			fwrite(encodedFrames[frameIdx].AccumulatedSyndrome[j], sizeof(unsigned char), encodedFrames[frameIdx].SyndromeByteLength, fp);
		}

		outputGOP(begin, frameIdx, fp);
		outputGOP(frameIdx, end, fp);
	}
}


// output encoded bitstream:
// resolution (0:QCIF, 1:CIF) + Qindex + IntraQP + GOP + frame number + deblocking filter mode
// + each frame< AC range, binary decision map, CRC, Syndrome >
void writeWZbitstream(char *fileName){
	FILE *fp = fopen(fileName, "wb+");
	unsigned char buffer;
	//output resolution (1 byte)
	buffer = (codingOption.FrameWidth == 176) ? 0 : 1;	//(0:QCIF, 1:CIF)
	fwrite(&buffer, sizeof(unsigned char), 1, fp);
	//output Qindex (1 byte)
	fwrite(&(codingOption.Qindex), sizeof(unsigned char), 1, fp);
	//output Intra block QP (1 byte)
    fwrite(&(codingOption.IntraQP), sizeof(unsigned char), 1, fp);
	//output GOP size (1 byte)
	fwrite(&(codingOption.GOP), sizeof(unsigned char), 1, fp);
	//output frame number (2 bytes)
	buffer = (unsigned char)((codingOption.FramesToBeEncoded)>>8);
	fwrite(&buffer, sizeof(unsigned char), 1, fp);
	buffer = (unsigned char)(codingOption.FramesToBeEncoded);
	fwrite(&buffer, sizeof(unsigned char), 1, fp);
	//output deblocking filter mode (1 byte)
	fwrite(&(codingOption.DeblockingMode), sizeof(unsigned char), 1, fp);

	for(int i=0; i<codingOption.FramesToBeEncoded; i+=codingOption.GOP){
		//for each GOP
		int begin = i;
		int end = i + codingOption.GOP;
		if(end >= codingOption.FramesToBeEncoded)
			end = codingOption.FramesToBeEncoded - 1;
		outputGOP(begin, end, fp);
	}
	fclose(fp);
}
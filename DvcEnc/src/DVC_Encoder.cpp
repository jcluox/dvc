#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "transform.h"
#include "quantization.h"
#include "bitplane.h"
#include "LdpcaEncoder.h"	//¤p¤p¥Õ
#include "skipBlock.h"
#include "FastMCFI.h"	//Boris
#include "intra.h"
#include "modeSelection.h"
#include "reconstruction.h"
#include "deblock.h"

void encodeGOP(int begin, int end, int *wzCount){
	int frameIdx = (begin + end) / 2;
	if(frameIdx > begin){
		//encode WZ frame
		printf("encode WZ frame %d...\n", frameIdx);

		encodedFrames[frameIdx].isWZ = 1;

		//skip Block selection
		encodedFrames[frameIdx].skipMode = 0;
		memset(encodedFrames[frameIdx].skipBlockFlag, 0, sizeof(unsigned char) * Block88Num);
		if(codingOption.EnableSkipBlock == 1)
			findSkipBlock(oriFrames[frameIdx], oriFrames[begin], &(encodedFrames[frameIdx]));

		//transform
		int *transFrame = (int*)malloc(sizeof(int) * FrameSize);
		//forwardTransform(oriFrames[frameIdx], oriFrames[begin], transFrame, &(encodedFrames[frameIdx]));
		forwardTransform(oriFrames[frameIdx], transFrame, &(encodedFrames[frameIdx]));

		//quantization
		quantize(transFrame, &(encodedFrames[frameIdx]));

		//intra mode selection
		encodedFrames[frameIdx].intraMode = 0;
		memset(encodedFrames[frameIdx].intraBlockFlag, 0, sizeof(unsigned char) * Block88Num);
		if(codingOption.EnableIntraMode == 1){
			//side info creation
            unsigned char *sideInfoFrame = (unsigned char*)malloc(sizeof(unsigned char) * FrameSize);
            FMCFI(1, sideInfoFrame, frameIdx, begin, end, &(encodedFrames[frameIdx]));

			// intra/WZ mode selection      
            intra_selection(&(encodedFrames[frameIdx]), sideInfoFrame, frameIdx);

			// re-calculate parameters
            recal_decision(&(encodedFrames[frameIdx]));

			// reconstruction (for WZ block)
            int *sideTransFrame = (int*)malloc(sizeof(int) * FrameSize); 
			int *reconsTransFrame = (int*)malloc(sizeof(int) * FrameSize);
            forwardTransform_Pure(sideInfoFrame, sideTransFrame, &(encodedFrames[frameIdx]));	//transform side info
            reconstruction(reconsTransFrame, transFrame, sideTransFrame, alpha_perframe, &(encodedFrames[frameIdx]));	//unquantized WZ frame

			// inverse transform (copy skip block as well), H264 intra reconstruct yuv has been stored in reconsFrames at intra_selection
            inverseTransform(reconsTransFrame, reconsFrames[begin], reconsFrames[frameIdx], &(encodedFrames[frameIdx])); //pixel domain WZ frame

			if(encodedFrames[frameIdx].intraMode == 1){
                // intra encoding            
                real_intra(&(encodedFrames[frameIdx]), frameIdx);
				// write out intra bitstream to file (byte align, not all)
				fwrite(IntraBuffer.streamBuffer, sizeof(unsigned char), IntraBuffer.byte_pos, IntraFp);
            }
            IntraBuffer.byte_pos = 0;

			if(codingOption.DeblockingMode){
				//deblocking filter
				unsigned char *deblockframe = (unsigned char*)malloc(sizeof(unsigned char) * FrameSize);
                unsigned char *type_map = (unsigned char*)malloc(sizeof(unsigned char) * Block88Num);
                init_type_map(type_map, encodedFrames[frameIdx].intraBlockFlag, encodedFrames[frameIdx].skipBlockFlag);

                deblock(deblockframe, reconsFrames[frameIdx], sideTransFrame, reconsTransFrame, type_map, begin);

                memcpy(reconsFrames[frameIdx], deblockframe, sizeof(unsigned char) * FrameSize);
                free(type_map);
                free(deblockframe);
			}

			free(sideInfoFrame);
			free(reconsTransFrame);
		}		
		

		//bitplane extraction
		unsigned char **bitplanes = (unsigned char**)malloc(sizeof(unsigned char*) * QbitsNumPerBlk);
		for(int i=0; i<QbitsNumPerBlk; i++)
			bitplanes[i] = (unsigned char*)malloc(sizeof(unsigned char) * Block44Num);
		extractBitplane(transFrame, bitplanes, &(encodedFrames[frameIdx]));

		//ldpca encode
		int SyndromeByteLength;
		for(int i=0; i<QbitsNumPerBlk; i++)
			encodedFrames[frameIdx].AccumulatedSyndrome[i] = LdpcaEncoder(bitplanes[i], &SyndromeByteLength, &(encodedFrames[frameIdx].CRC[i]));
		encodedFrames[frameIdx].SyndromeByteLength = SyndromeByteLength;

		for(int i=0; i<QbitsNumPerBlk; i++)
			free(bitplanes[i]);
		free(bitplanes);
		free(transFrame);
		

		(*wzCount)++;
		encodeGOP(begin, frameIdx, wzCount);
		encodeGOP(frameIdx, end, wzCount);
	}
}

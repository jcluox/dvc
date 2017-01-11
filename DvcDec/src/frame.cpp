#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "global.h"
#include "error.h"
#include "reconstruction.h"

extern int LDPC_iterations;

void initFrameBuffer(){
	oriFrames = (unsigned char**)malloc(sizeof(unsigned char*) * codingOption.FramesToBeDecoded);
	decodedFrames = (DecodedFrame*)malloc(sizeof(DecodedFrame) * codingOption.FramesToBeDecoded);
	for(int i=0; i<codingOption.FramesToBeDecoded; i++){
		oriFrames[i] = (unsigned char*)malloc(sizeof(unsigned char) * FrameSize);
		decodedFrames[i].reconsFrame = (unsigned char*)malloc(sizeof(unsigned char) * FrameSize);
		decodedFrames[i].reconsFrame = (unsigned char*)malloc(sizeof(unsigned char) * FrameSize);
	}
	
	//sideInfo.initSideInfo = (int*)malloc(sizeof(int) * FrameSize);
	//sideInfo.newSideInfo = (int*)malloc(sizeof(int) * FrameSize);
	sideInfo.sideInfoFrame = (unsigned char*)malloc(sizeof(unsigned char) * FrameSize);
	sideInfo.residueFrame = (int*)malloc(sizeof(int) * FrameSize);
	sideInfo.reconsTransFrame = (int*)malloc(sizeof(int) * FrameSize);
	sideInfo.BlockMV = (MV*)malloc(sizeof(MV) * Block88Num);
	sideInfo.MSE = (double*)malloc(sizeof(double) * Block88Num);
	sideInfo.mcroBlockMV = (MV*)malloc(sizeof(MV) * Block1616Num);
	sideInfo.Holes = (Point*)malloc(sizeof(Point) * Block1616Num);
	sideInfo.alpha = (float*)malloc(sizeof(float) * FrameSize);
	//sideInfo.noiseDist = (float**)malloc(sizeof(float*) * Block44Num);
	SearchInfo1616 = (SearchInfo*)malloc(sizeof(SearchInfo) * Block1616Num);
}

void freeFrameBuffer(){
	for(int i=0; i<codingOption.FramesToBeDecoded; i++){
		free(oriFrames[i]);
		free(decodedFrames[i].reconsFrame);
		if(decodedFrames[i].hasReferenced == 1){
			free(decodedFrames[i].lowPassFrame);
			free(decodedFrames[i].interpolateFrame);
		}
		if(decodedFrames[i].isWZFrame == 1){
			for(int j=0; j<QbitsNumPerBlk; j++)
				free(encodedFrames[i].AccumulatedSyndrome[j]);
			free(encodedFrames[i].AccumulatedSyndrome);
			free(encodedFrames[i].CRC);
			free(encodedFrames[i].BlockModeFlag);
			if(encodedFrames[i].intraMode)
				free(encodedFrames[i].blockInfo);
		}
	}
	free(oriFrames);

	//free(sideInfo.initSideInfo);
	//free(sideInfo.newSideInfo);
	free(sideInfo.sideInfoFrame);
	free(sideInfo.residueFrame);
	free(sideInfo.reconsTransFrame);
	free(sideInfo.BlockMV);
	free(sideInfo.MSE);
	free(sideInfo.mcroBlockMV);
	free(sideInfo.Holes);
	free(sideInfo.alpha);
	//free(sideInfo.noiseDist);
	free(SearchInfo1616);

	free(encodedFrames);
	free(decodedFrames);
}

void readSequence(char *fileName, unsigned char **frames){
	FILE *fp = fopen(fileName, "rb");
	if(fp == NULL){
		sprintf(errorText, "File %s doesn't exist\n", fileName);
		errorMsg(errorText);
	}

	int uvSize = FrameSize/2;
	unsigned char *buf = (unsigned char *)malloc(sizeof(unsigned char) * uvSize); 
	for(int i=0; i<codingOption.FramesToBeDecoded; i++){
		//read y
		if(fread(frames[i], 1, FrameSize, fp) != FrameSize){
			sprintf(errorText, "Input sequence %s is not enough", fileName);
			errorMsg(errorText);
		}
		
		//read u,v
		fread(buf, 1, uvSize, fp);
	}	
	free(buf);
	fclose(fp);
}

void outputSequence(char *fileName){
	//unsigned char *uvbuffer = (unsigned char*)malloc(sizeof(unsigned char) * FrameSize / 2);
	//memset(uvbuffer, 0, FrameSize/2);
	FILE *fp = fopen(fileName, "wb+");
	int end;
	if(codingOption.FramesToBeDecoded-1 == codingOption.endFrame)
		end = codingOption.endFrame;
	else
		end = codingOption.endFrame-1;
	for(int i=codingOption.startFrame; i<=end; i++){
	//for(int i=0;i<codingOption.FramesToBeDecoded;i++){
		fwrite(decodedFrames[i].reconsFrame, sizeof(unsigned char), FrameSize, fp);

		//uv is set to zero
		//fwrite(uvbuffer, sizeof(unsigned char), FrameSize/2, fp);
	}
	fclose(fp);
	//free(uvbuffer);
}

double psnr(unsigned char *frame1, unsigned char *frame2){
	double mse = 0.0;
	for(int i=0; i<FrameSize; i++){
		double diff = (double)frame1[i] - (double)frame2[i];
		mse += diff * diff;
	}
	mse /= (double)FrameSize;

	return ( mse == 0.0 ? 0.0 : (10.0 * log10(255.0 * 255.0 / mse)) );
}

void write_bmpheader(unsigned char *bitmap, int offset, int bytes, int value){
	int i;
	for(i=0; i<bytes; i++)
		bitmap[offset+i] = (value >> (i<<3)) & 0xFF;
}

unsigned char *convertToBmp(unsigned char *inputImg, int width, int height, int *ouputSize){
	/*create a bmp format file*/
	int bitmap_x = (int)ceil((double)width*3/4) * 4;	
	unsigned char *bitmap = (unsigned char*)malloc(sizeof(unsigned char)*height*bitmap_x + 54);
	
	bitmap[0] = 'B';
	bitmap[1] = 'M';
	write_bmpheader(bitmap, 2, 4, height*bitmap_x+54); //whole file size
	write_bmpheader(bitmap, 0xA, 4, 54); //offset before bitmap raw data
	write_bmpheader(bitmap, 0xE, 4, 40); //length of bitmap info header
	write_bmpheader(bitmap, 0x12, 4, width); //width
	write_bmpheader(bitmap, 0x16, 4, height); //height
	write_bmpheader(bitmap, 0x1A, 2, 1);
	write_bmpheader(bitmap, 0x1C, 2, 24); //bit per pixel
	write_bmpheader(bitmap, 0x1E, 4, 0); //compression
	write_bmpheader(bitmap, 0x22, 4, height*bitmap_x); //size of bitmap raw data
	for(int i=0x26; i<0x36; i++)
		bitmap[i] = 0;

	int k=54;
	for(int i=height-1; i>=0; i--){
		int j;
		for(j=0; j<width; j++){
			int index = i*width+j;
			for(int l=0; l<3; l++)
				bitmap[k++] = inputImg[index];
		}
		j*=3;
		while(j<bitmap_x){
			bitmap[k++] = 0;
			j++;
		}
	}
	
	*ouputSize = k;
	return bitmap;
}

void saveToBmp(unsigned char *inputImg, int width, int height, char *outputFileName){
	int size;
	unsigned char *bmp = convertToBmp(inputImg, width, height, &size);
	FILE *fp = fopen(outputFileName, "wb+");
	if(fp == NULL){
		sprintf(errorText, "Could not open file: %s", outputFileName);
		errorMsg(errorText);
	}
	fwrite(bmp, 1, size, fp);
	fclose(fp);
	free(bmp);
}

void printBlock44(int *blk){
	for(int i=0; i<4; i++){
		for(int j=0; j<4; j++)
			printf("%4d", blk[i*4+j]);
		printf("\n");
	}
#ifdef _WIN32
	system("pause");
#endif
}

//­×§ï¦Ûlyman
// Read the info of key frame (from h264 output)
// variable num is the number of the frame
void read_one_frame_info(int frameNo, int *y_bits, double *y_psnr, char *fileName)
{
	
	int frameno = 0, qp=0, uv_bits=0;
	char temp[50];
	FILE *fp;

	fp = fopen(fileName, "r"); // Lyman 20100119
	if(fp == NULL){
		sprintf(errorText, "File %s not found", fileName);
		errorMsg(errorText);
	}

	while(!feof(fp))
	{
		fscanf(fp,"%s", temp);
		if(strcmp(temp, "FrameNo:") == 0)
		{
			fscanf(fp, "%s", temp); // get frame number
			frameno = atoi(temp);

			if(frameno == frameNo)
			{
				break;
			}
		}
	}

	if(feof(fp))
	{
		sprintf(errorText, "In function read_one_frame_info: frame number %d not found", frameNo);
		errorMsg(errorText);
	}


	fscanf(fp,"%s",temp);
	fscanf(fp,"%s",temp); // get Y_bits
	*y_bits = atoi(temp);

	fscanf(fp,"%s",temp);
	fscanf(fp,"%s",temp); // get UV_bits

	fscanf(fp,"%s",temp);
	fscanf(fp,"%s",temp); // get QP

	fscanf(fp,"%s",temp);
	fscanf(fp,"%s",temp); // get Y_psnr
	*y_psnr = (float)atof(temp);

	fclose(fp);
}

void reportDecodeInfo(char *fileName, double totalDecodeTime){
	double totalWZBits = 0.0, totalKeyBits = 0.0;
	double totalWZPSNR = 0.0, totalKeyPSNR = 0.0;
	double wzCount = 0.0;
	double keyCount = 0.0;
	FILE *fp = fopen(fileName, "w+");
	for(int i=codingOption.startFrame; i<=codingOption.endFrame; i++){
	//for(int i=0;i<codingOption.FramesToBeDecoded;i++){
		double bits = (double)(decodedFrames[i].bitRate);
		double psnr = decodedFrames[i].PSNR;
		if(decodedFrames[i].isWZFrame == 1){
			totalWZBits += bits;
			totalWZPSNR += psnr;
			wzCount++;
			fprintf(fp, "FrameNo: %d, Bits: %.0lf, PSNR: %.2lf (WZ), Decoding time: %.2lf ms\n", i, bits, psnr, decodedFrames[i].decodeTime);
		}
		else{
			totalKeyBits += bits;
			totalKeyPSNR += psnr;
			keyCount++;
			fprintf(fp, "FrameNo: %d, Bits: %.0lf, PSNR: %.2lf (KY)\n", i, bits, psnr);
		}
	}
	fprintf(fp, "\n");
	char text[100];	
	double avgBits = (totalKeyBits+totalWZBits) / (keyCount+wzCount) / 1000.0;
	sprintf(text, "File Size: %.2lf KB\n", (totalKeyBits+totalWZBits)/8.0/1024.0);
	printf("%s", text);	fprintf(fp, "%s", text);
	sprintf(text, "Average WZ Bits(k): %.2lf, Average WZ PSNR: %.2lf (%.0lf frames)\n", totalWZBits / wzCount / 1000.0, totalWZPSNR / wzCount, wzCount);
	printf("%s", text);	fprintf(fp, "%s", text);
	sprintf(text, "Average KY Bits(k): %.2lf, Average KY PSNR: %.2lf (%.0lf frames)\n", totalKeyBits / keyCount / 1000.0, totalKeyPSNR / keyCount, keyCount);
	printf("%s", text);	fprintf(fp, "%s", text);
	sprintf(text, "Average Bits(k): %.2lf, Average PSNR: %.2lf (%.0lf frames)\n", avgBits, (totalKeyPSNR+totalWZPSNR) / (keyCount+wzCount), keyCount+wzCount);
	printf("%s", text);	fprintf(fp, "%s", text);
	sprintf(text, "BitRate (including key frame): %.2lf kbps\nTotal decoding time: %.2lf sec\n", avgBits * (double)codingOption.FrameRate, totalDecodeTime);
	printf("%s", text);	fprintf(fp, "%s", text);
	sprintf(text, "Average WZ decoding speed: %.2lf fps\n", wzCount / totalDecodeTime);
	printf("%s", text);	fprintf(fp, "%s", text);

	//printf("bit2byte = %.2lf ms\n",byte2bitTime);
	#ifdef SHOW_TIME_INFO
		sprintf(text, "\nTime information of each component:\n");
		printf("%s", text);	fprintf(fp, "%s", text);
		sprintf(text, "Side Info Creation: %.2lf sec\n", Total_sideCreate_time/1000.0);
		printf("%s", text);	fprintf(fp, "%s", text);
		sprintf(text, "Overcomplete Transform: %.2lf sec\n", Total_overT_time/1000.0);
		printf("%s", text);	fprintf(fp, "%s", text);
		sprintf(text, "Correlation Noise Modeling: %.2lf sec\n", Total_CNM_time/1000.0);
		printf("%s", text);	fprintf(fp, "%s", text);
		sprintf(text, "Conditional Bit Prob Compute: %.2lf sec\n", Total_condBit_time/1000.0);
		printf("%s", text);	fprintf(fp, "%s", text);
		sprintf(text, "Update Side Info: %.2lf sec\n", Total_updateSI_time/1000.0);
		printf("%s", text);	fprintf(fp, "%s", text);
		sprintf(text, "Motion Learning: %.2lf sec\n", Total_ML_time/1000.0);
		printf("%s", text);	fprintf(fp, "%s", text);
		sprintf(text, "Others: %.2lf sec\n", totalDecodeTime-(Total_sideCreate_time+Total_overT_time+Total_CNM_time+Total_condBit_time+Total_updateSI_time+Total_ldpca_time+Total_ML_time)/1000.0);
		printf("%s", text);	fprintf(fp, "%s", text);
		sprintf(text, "Ldpca Decode: %.2lf sec\n", Total_ldpca_time/1000.0);
		printf("%s", text);	fprintf(fp, "%s", text);
		//printf("number of iterations %d\n",LDPC_iterations);
		#ifdef LDPC_ANALYSIS
			for(int i=0;i<numCodes;i++){
				int step = 0;
				LdpcInfo info = ldpcInfo[i];
				double sum = info.toGPU_time + info.Row_time + 
							 info.Col_time   + info.BpCheck_time + 
							 info.toCPU_time + info.CPUloop_time + 
							 info.checkResult_time + info.InverseMatrix_time;
				sprintf(text, "\n\tLDPCA Average Time for ladder[%d] %dbits (count=%d)\n",i, (i+2)*24,info.count);
				printf("%s", text);	fprintf(fp, "%s", text);
				if(i == numCodes-1){
					sprintf(text, "Inverse Matrix Multiplication Time:\t\t%.2lf ms(%.2lf%%)\n", info.InverseMatrix_time/(double)(info.count), info.InverseMatrix_time/sum*100);
					printf("%s", text);	fprintf(fp, "%s", text);
					break;
				}
				sprintf(text, "Step%d CPU->GPU:\t\t%.2lf ms(%.2lf%%)\n", step++, info.toGPU_time/(double)(info.count), info.toGPU_time/sum*100);
				printf("%s", text);	fprintf(fp, "%s", text);
				sprintf(text, "Step%d Horizontal:\t%.2lf ms(%.2lf%%)\n", step++, info.Row_time/(double)(info.count), info.Row_time/sum*100);
				printf("%s", text);	fprintf(fp, "%s", text);
				sprintf(text, "Step%d BP Check:\t\t%.2lf ms(%.2lf%%)\n", step++, info.BpCheck_time/(double)(info.count), info.BpCheck_time/sum*100); //?
				printf("%s", text);	fprintf(fp, "%s", text);
				sprintf(text, "Step%d get decoded:\t%.2lf ms(%.2lf%%)\n", step++, info.checkResult_time/(double)(info.count), info.checkResult_time/sum*100); //?
				printf("%s", text);	fprintf(fp, "%s", text);
			}
		#endif
	#endif

	fclose(fp);
}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "global.h"
#include "error.h"
#include "reconstruction.h"
#include "omp.h"

void displayDecodeParamters(){
	printf("******************************************************\n");
	printf("*               Decoder Parameters                   *\n");
	printf("******************************************************\n");
	printf("Original sequence: %s\n", codingOption.OriYuvFile);
	printf("Key frame sequence: %s\n", codingOption.KeyFrameYuvFile);
	printf("Key frame information: %s\n", codingOption.KeyFrameInfo);
	printf("FramesToBeDecoded: %d\n", codingOption.FramesToBeDecoded);
	printf("FrameRate: %d fps\n", codingOption.FrameRate);
	printf("SequenceSize: %s\n", (codingOption.FrameWidth == 176) ? "QCIF (176x144)" : "CIF (352x288)");
	printf("GOP size: %d\n", codingOption.GOP);
	printf("Qindex: %d\n", codingOption.Qindex);
	printf("Quantization table:\n");
	for(int i=0;i<16;i++){
		printf("%3d", Qtable[i]);
		if((i+1)%4 == 0)
			printf("\n");
	}
	printf("RDFile: %s\n", codingOption.RDFile);
	printf("Decoded sequence: %s\n", codingOption.OutputDecodedYuvFile);
	switch(codingOption.ParallelMode){
		case 0:
			printf("Parallel Mode: %d (sequential)\n", codingOption.ParallelMode);
			break;
		case 1:
			printf("Parallel Mode: %d (OpenMP only)\n", codingOption.ParallelMode);
			break;
		case 2:
			printf("Parallel Mode: %d (CUDA only)\n", codingOption.ParallelMode);
			break;
		case 3:
			printf("Parallel Mode: %d (OpenMP + CUDA)\n", codingOption.ParallelMode);
			break;
		default:
			break;
	}
	printf("Use %d %s (CPU)\n", codingOption.UseCoreNum, (codingOption.UseCoreNum==1) ? "core" : "cores");
	printf("******************************************************\n");
}

int parameterNameToMapIndex(const char * str){
	for(int i=0;i< PARAMETERNUM;i++){
		if(STRCASECMP(parameters[i], str) == 0)
			return i;
	}
	return -1;
}

void overrideParameter(char *cmd, char *argFlag){
	char *parameterName, *parameterValue;
	parameterName = &(cmd[1]);
	int argLen = (int)strlen(parameterName);
	for(int i=0; i<argLen; i++)
		if(parameterName[i] == '='){
			parameterName[i] = '\0';
			parameterValue = &(parameterName[i+1]);
			break;
		}
	int index = parameterNameToMapIndex(parameterName);
	if(index < 0){
		sprintf(errorText, "Invalid flag -%s", parameterName);
		errorMsg(errorText);
	}
	else{
		argFlag[index] = 1;
		switch(index){
			case 0:
				sprintf(codingOption.OriYuvFile, "%s", parameterValue);				
				break;
			case 1:
				sprintf(codingOption.WZBitStreamFile, "%s", parameterValue);
				break;
			case 2:
				sprintf(codingOption.KeyFrameYuvFile, "%s", parameterValue);				
				break;
			case 3:
				sprintf(codingOption.KeyFrameInfo, "%s", parameterValue);
				break;
			case 4:
				sprintf(codingOption.RDFile, "%s", parameterValue);
				break;
			case 5:
				sprintf(codingOption.OutputDecodedYuvFile, "%s", parameterValue);
				break;
			case 6:
				codingOption.FrameRate = atoi(parameterValue);
				break;
			case 7:
				codingOption.ParallelMode = atoi(parameterValue);
				break;
			case 8:
				codingOption.UseCoreNum = atoi(parameterValue);			
				break;
			case 9:
				sprintf(codingOption.IntraBitStreamFile, "%s", parameterValue);
				break;
			default:
				break;
		}
	}
}

static char* helpStr = "Usage: DvcDec.exe -d <Configure File> [-ParameterName=ParameterValue]\n\
			========List of ParameterName========\n\
	-OriginalSequence	Original sequence (used for PSNR calculation) (input file)\n\
	-WZBitstream		WZ bitstream file (input file)\n\
	-IntraBitstream		Intra block bitstream file (input file)\n\
	-KeyFrameSequence	reconstruct yuv by h264 (input file)\n\
	-KeyFrameInfo		key frame infomation from h264 (input file)\n\
	-RDFile			report rate and PSNR file (output file)\n\
	-OutputSequences	Decoded file (without extension) (output file)\n\
	-FrameRate		Frame rate (used for rate calculation)\n\
	-ParallelMode		0: sequential, 1: OpenMP only, 2: CUDA only, 3: OpenMP + CUDA\n\
	-GPUid			specify which GPU card used for computing\n\
	-startFrame		first frame for decoding(start from 0)\n\
	-endFrame		last frame for decoding\n\
	-UseCoreNum		number of CPU used by OpenMP (ParallelMode: 1 or 3)\n";

void readConfigure(int argc, char **argv){

	if(argc < 3)
		errorMsg(helpStr);
	char argFlag[PARAMETERNUM] = {0};
	char configFile[300];
	codingOption.startFrame = -1;
	codingOption.endFrame = -1;
	codingOption.GPUid = 0;
	for(int i=1; i<argc; i++){
		if(STRCASECMP(argv[i], "-d") == 0){
			sprintf(configFile, "%s", argv[i+1]);
		}
		else if(argv[i][0] == '-'){
			//====JSRF 設定 start/end frame && GPUid寫在這裡，原本的太過於複雜 寫在這裡算是偷懶===========
			if(strstr(argv[i], "-GPUid=") != 0){
				sscanf(argv[i], "-GPUid=%d", &codingOption.GPUid);
				//printf("GPU id = %d\n",codingOption.GPUid);getchar();
			}
			else if(strstr(argv[i], "-startFrame=") != 0){
				sscanf(argv[i], "-startFrame=%d", &codingOption.startFrame);
				//printf("startFrame = %d\n",codingOption.startFrame);getchar();
			}
			else if(strstr(argv[i], "-endFrame=") != 0){
				sscanf(argv[i], "-endFrame=%d", &codingOption.endFrame);
				//printf("endFrame = %d\n",codingOption.endFrame);getchar();
			}
			else
				overrideParameter(argv[i], argFlag);
		}
	}
	
	FILE *fp = fopen(configFile, "r");
	if(fp == NULL){	
		sprintf(errorText, "Configure file \"%s\" doesn't exist", configFile);
		errorMsg(errorText);
	}
	printf("Configure file name: %s\n", configFile);

	//以下讀configure檔的code是從h264 reference software抄過來的
	fseek (fp, 0, SEEK_END);
	long fileSize = ftell(fp);
	fseek (fp, 0, SEEK_SET);

	char *buf;
	if( ( buf = (char*)malloc (sizeof(char)*(fileSize+1)) ) == NULL){		
		errorMsg("Could not allocate memory: GetConfigFileContent: buf");
	}

	fileSize = (long)fread (buf, sizeof(char), fileSize, fp);
	
	buf[fileSize] = '\0';
	fclose(fp);

	char *items[100];  
	int item = 0;
	int InString = 0, InItem = 0;
	char *p = buf;
	char *bufend = &buf[fileSize];  

	while (p < bufend){
		switch (*p){
			case 13:
				p++;
				break;
			case '#':                 // Found comment
				*p = '\0';              // Replace '#' with '\0' in case of comment immediately following integer or string
				while (*p != '\n' && p < bufend)  // Skip till EOL or EOF, whichever comes first
					p++;
				InString = 0;
				InItem = 0;
				break;
			case '\n':
				InItem = 0;
				InString = 0;
				*p++='\0';
				break;
			case ' ':
			case '\t':              // Skip whitespace, leave state unchanged
				if (InString)
					p++;
				else{               // Terminate non-strings once whitespace is found
					*p++ = '\0';
					InItem = 0;
				}
				break;
			case '"':               // Begin/End of String
				*p++ = '\0';
				if (!InString){
					items[item++] = p;
					InItem = ~InItem;
				}
				else
					InItem = 0;
				InString = ~InString; // Toggle
				break;
			default:
				if (!InItem){
					items[item++] = p;
					InItem = ~InItem;
				}
				p++;
		}
	}

	item--;	

	int count = 0;
	for(int i=0; i<item; i+=3){
		int index = parameterNameToMapIndex(items[i]);
		if(index < 0){
			sprintf(errorText, "Parsing error in config file: Parameter Name '%s' not recognized.", items[i]);
			errorMsg(errorText);
		}	
		if (STRCASECMP("=", items[i+1]) < 0){			
			errorMsg("Parsing error in config file: '=' expected as the second token in each line.");
		}
		
		switch(index){
			case 0:
				if(argFlag[index] == 0)
					sprintf(codingOption.OriYuvFile, "%s", items[i+2]);				
				break;
			case 1:
				if(argFlag[index] == 0)
					sprintf(codingOption.WZBitStreamFile, "%s", items[i+2]);
				break;
			case 2:
				if(argFlag[index] == 0)
					sprintf(codingOption.KeyFrameYuvFile, "%s", items[i+2]);				
				break;
			case 3:
				if(argFlag[index] == 0)
					sprintf(codingOption.KeyFrameInfo, "%s", items[i+2]);
				break;
			case 4:
				if(argFlag[index] == 0)
					sprintf(codingOption.RDFile, "%s", items[i+2]);
				break;
			case 5:
				if(argFlag[index] == 0)
					sprintf(codingOption.OutputDecodedYuvFile, "%s", items[i+2]);
				break;
			case 6:
				if(argFlag[index] == 0)
					codingOption.FrameRate = atoi(items[i+2]);
				break;
			case 7:
				if(argFlag[index] == 0)
					codingOption.ParallelMode = atoi(items[i+2]);
				#ifndef CUDA
				if(codingOption.ParallelMode < 0 || codingOption.ParallelMode > 1)
					errorMsg("ParallelMode must be 0 or 1 (the compiled binary \"DvcDec.exe\" does not include CUDA program)");
				#else
				if(codingOption.ParallelMode < 0 || codingOption.ParallelMode > 3)
					errorMsg("ParallelMode must be 0 ~ 3");
				#endif
				break;
			case 8:
				if(argFlag[index] == 0)
					codingOption.UseCoreNum = atoi(items[i+2]);
				if(codingOption.ParallelMode==1 || codingOption.ParallelMode==3){
					int procNum = omp_get_num_procs();
					if(codingOption.UseCoreNum > procNum)
						codingOption.UseCoreNum = procNum;
					//number of cores used by OpenMP
					omp_set_num_threads(codingOption.UseCoreNum);
				}
				else
					codingOption.UseCoreNum = 1;
				break;
			case 9:
				if(argFlag[index] == 0)
					sprintf(codingOption.IntraBitStreamFile, "%s", items[i+2]);
				break;
			default:
				break;
		}

		count++;
	}

	if(count != PARAMETERNUM)
		errorMsg("Parsing error in config file: Too few parameters");

	free(buf);
}

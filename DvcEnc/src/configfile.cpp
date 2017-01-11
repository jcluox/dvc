#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "global.h"
#include "error.h"
#include "quantization.h"

void displayEncodeParamters(){
	printf("******************************************************\n");
	printf("*               Encoder Parameters                   *\n");
	printf("******************************************************\n");
	printf("InputFile: %s\n", codingOption.InputFile);
	printf("FramesToBeEncoded: %d\n", codingOption.FramesToBeEncoded);
	printf("SequenceSize: %s\n", (codingOption.FrameWidth == 176) ? "QCIF (176x144)" : "CIF (352x288)");
	printf("GOP size: %d\n", codingOption.GOP);
	printf("Qindex: %d\n", codingOption.Qindex);
	printf("Quantization table:\n");
	for(int i=0;i<16;i++){
		printf("%3d", Qtable[i]);
		if((i+1)%4 == 0)
			printf("\n");
	}
	printf("SkipBlock: %s\n", (codingOption.EnableSkipBlock == 0)? "Disable" : "Enable");
	printf("IntraMode: %s\n", (codingOption.EnableIntraMode == 0)? "Disable" : "Enable");
	if(codingOption.EnableIntraMode == 1){
		printf("KeyFrameSequence: %s\n", codingOption.KeyFrameYuvFile);
		printf("IntraQP: %d\n", codingOption.IntraQP);
		printf("IntraOutputFile: %s\n", codingOption.IntraOutputFile);
	}
	printf("DeblockingFilter: %s\n", (codingOption.DeblockingMode == 0)? "Disable" : "Enable");
	printf("WZOutputFile: %s\n", codingOption.WZOutputFile);
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
				sprintf(codingOption.InputFile, "%s", parameterValue);				
				break;
			case 1:
				codingOption.FramesToBeEncoded = atoi(parameterValue);
				break;
			case 2:				
				if(STRCASECMP(parameterValue, "QCIF") == 0){
					codingOption.FrameWidth = 176;
					codingOption.FrameHeight = 144;
				}
				else if(STRCASECMP(parameterValue, "CIF") == 0){
					codingOption.FrameWidth = 352;
					codingOption.FrameHeight = 288;
				}
				else{
					sprintf(errorText, "Parsing error in config file: not support SequenceSize = %s", parameterValue);
					errorMsg(errorText);
				}				
				break;
			case 3:
				codingOption.GOP = atoi(parameterValue);
				break;
			case 4:
				codingOption.Qindex = atoi(parameterValue);
				break;
			case 5:				
				sprintf(codingOption.WZOutputFile, "%s", parameterValue);			
				break;
			case 6:
				codingOption.EnableSkipBlock = atoi(parameterValue);
				break;
			case 7:
				codingOption.EnableIntraMode = atoi(parameterValue);
				break;
			case 8:
				codingOption.IntraQP = (unsigned char)atoi(parameterValue);
				break;
			case 9:
				codingOption.DeblockingMode = (unsigned char)atoi(parameterValue);
				break;
			case 10:
				sprintf(codingOption.KeyFrameYuvFile, "%s", parameterValue);
				break;
			case 11:
				sprintf(codingOption.IntraOutputFile, "%s", parameterValue);
				break;
			default:
				break;
		}
	}
}

void readConfigure(int argc, char **argv){

	if(argc < 3)
		errorMsg("Usage: DvcEnc.exe -d <Configure File> [-ParameterName=ParameterValue]");
	char argFlag[PARAMETERNUM] = {0};
	char configFile[300];
	for(int i=1; i<argc; i++){
		if(STRCASECMP(argv[i], "-d") == 0){
			sprintf(configFile, "%s", argv[i+1]);
		}
		else if(argv[i][0] == '-'){
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
		if (STRCASECMP ("=", items[i+1]) < 0){			
			errorMsg("Parsing error in config file: '=' expected as the second token in each line.");
		}
		
		switch(index){
			case 0:
				if(argFlag[index] == 0)
					sprintf(codingOption.InputFile, "%s", items[i+2]);				
				break;
			case 1:
				if(argFlag[index] == 0)
					codingOption.FramesToBeEncoded = atoi(items[i+2]);
				break;
			case 2:
				if(argFlag[index] == 0){
					if(STRCASECMP(items[i+2], "QCIF") == 0){
						codingOption.FrameWidth = 176;
						codingOption.FrameHeight = 144;
					}
					else if(STRCASECMP(items[i+2], "CIF") == 0){
						codingOption.FrameWidth = 352;
						codingOption.FrameHeight = 288;
					}
					else{
						sprintf(errorText, "Parsing error in config file: not support SequenceSize = %s", items[i+2]);
						errorMsg(errorText);
					}
				}
				FrameSize = codingOption.FrameWidth * codingOption.FrameHeight;
				Block44Num = FrameSize / BLOCKSIZE44;
				Block88Num = FrameSize / BLOCKSIZE88;
				Block44Height = codingOption.FrameHeight / 4;
				Block44Width = codingOption.FrameWidth / 4;
				Block88Height = codingOption.FrameHeight / 8;
				Block88Width = codingOption.FrameWidth / 8;	
				break;
			case 3:
				if(argFlag[index] == 0)
					codingOption.GOP = atoi(items[i+2]);
				break;
			case 4:
				if(argFlag[index] == 0)
					codingOption.Qindex = atoi(items[i+2]);
				Qtable = QuantizationTable[codingOption.Qindex];
				QtableBits = QuantizationTableBits[codingOption.Qindex];
				for(int j=0; j<BLOCKSIZE44; j++)
					QbitsNumPerBlk += QtableBits[j];
				for(int j=1; j<BLOCKSIZE44; j++){
					if(Qtable[j] > 0)
						ACLevelShift[j] = (Qtable[j] - 2) / 2;
				}
				break;
			case 5:
				if(argFlag[index] == 0)
					sprintf(codingOption.WZOutputFile, "%s", items[i+2]);
				fp = fopen(codingOption.WZOutputFile, "wb+");
				if(fp == NULL){
					sprintf(errorText, "Cound not open output file: %s", codingOption.WZOutputFile);
					errorMsg(errorText);
				}
				fclose(fp);
				break;
			case 6:
				if(argFlag[index] == 0)
					codingOption.EnableSkipBlock = atoi(items[i+2]);
				if(codingOption.EnableSkipBlock!=0 && codingOption.EnableSkipBlock!=1)
					errorMsg("SkipBlock must be 0 or 1 (0: disable, 1: enable)");
				break;
			case 7:
				if(argFlag[index] == 0)
					codingOption.EnableIntraMode = atoi(items[i+2]);
				if(codingOption.EnableIntraMode!=0 && codingOption.EnableIntraMode!=1)
					errorMsg("IntraMode must be 0 or 1 (0: disable, 1: enable)");
				break;
			case 8:
				if(argFlag[index] == 0)
					codingOption.IntraQP = (unsigned char)atoi(items[i+2]);
				break;
			case 9:
				if(argFlag[index] == 0)
					codingOption.DeblockingMode = (unsigned char)atoi(items[i+2]);
				if(codingOption.DeblockingMode!=0 && codingOption.DeblockingMode!=1)
					errorMsg("DeblockingFilter must be 0 or 1 (0: disable, 1: enable)");
				break;
			case 10:
				if(argFlag[index] == 0)
					sprintf(codingOption.KeyFrameYuvFile, "%s", items[i+2]);
				break;
			case 11:
				if(argFlag[index] == 0)
					sprintf(codingOption.IntraOutputFile, "%s", items[i+2]);
				break;
			default:
				break;
		}

		count++;
	}

	if(count != PARAMETERNUM)
		errorMsg("Parsing error in config file: Too few parameters");

	free(buf);

	displayEncodeParamters();

}
#include <stdio.h>
#include <stdlib.h>
#include "global.h"
#include "error.h"
#include "ReadLdpcaLadderFile_LdpcaEncoder.h"

int EnCodeLength, EnCodeEdges;
int* EpJC;
int* EpIrRead;

void ReadLadder_encoder(char* LadderFileName)
{	
	int k;	

	FILE* pLadderFile = fopen(LadderFileName, "r");
	if(pLadderFile == NULL){
		sprintf(errorText, "File %s doesn't exist", LadderFileName);
		errorMsg(errorText);
	}
	
    fscanf(pLadderFile, "%d", &EnCodeLength);
    fscanf(pLadderFile, "%d", &EnCodeEdges);    
      
	EpJC = (int *)malloc(sizeof(int) * (EnCodeLength+1));

	for (k = 0; k < EnCodeLength+1; k++)
        fscanf(pLadderFile, "%d", EpJC+k);	

	EpIrRead = (int *)malloc(sizeof(int) * EnCodeEdges);
	
	for (k = 0; k < EnCodeEdges; k++)
		fscanf(pLadderFile, "%d", EpIrRead+k);
		
	fclose(pLadderFile);	
}

void freeLdpcaParam(){
	free(EpJC);
	free(EpIrRead);
}
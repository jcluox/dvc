#ifndef READLDPCALADDERFILE_LDPCAENCODER_H
#define READLDPCALADDERFILE_LDPCAENCODER_H

//�p�p��
extern int EnCodeLength, EnCodeEdges;
extern int* EpJC;
extern int* EpIrRead;

void ReadLadder_encoder(char* LadderFileName);
void freeLdpcaParam();

#endif
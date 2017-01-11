#ifndef READLDPCALADDERFILE_LDPCAENCODER_H
#define READLDPCALADDERFILE_LDPCAENCODER_H

//¤p¤p¥Õ
extern int EnCodeLength, EnCodeEdges;
extern int* EpJC;
extern int* EpIrRead;

void ReadLadder_encoder(char* LadderFileName);
void freeLdpcaParam();

#endif
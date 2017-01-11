#ifndef WZBITSTREAM_H
#define WZBITSTREAM_H

#include "stdio.h"

void readGOP(int begin, int end, FILE* fp);
void readWZBitstream(char *fileName);

#endif
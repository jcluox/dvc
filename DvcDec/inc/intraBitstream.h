#ifndef INTRABITSTREAM_H
#define INTRABITSTREAM_H

#include <stdio.h>

void readIntraBitstream(char *fileName);
void init_IntraBuffer(FILE *fp);
void free_IntraBuffer();

#endif
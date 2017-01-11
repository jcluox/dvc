#ifndef WZBITSTREAM_H
#define WZBITSTREAM_H

void outputGOP(int begin, int end, FILE* fp);
//output encoded bitstream (resolution (0:QCIF, 1:CIF) + Qindex + IntraQP + GOP + frame number + each frame AC range, skip block index, CRC, Syndrome)
void writeWZbitstream(char *fileName);

#endif
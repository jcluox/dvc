#ifndef LDPCAENCODER_H
#define LDPCAENCODER_H

unsigned char* BitToByte(unsigned char *Bits, int BitLength, int *numBytes);
unsigned char* ByteToBit(unsigned char *Bytes, int ByteLength, int SourceLength);
unsigned char* LdpcaEncoder(unsigned char *source, int* SyndromeByteLength, unsigned char *crc);

#endif

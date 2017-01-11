#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "global.h"
#include "crc8.h"
#include "error.h"
#include "ReadLdpcaLadderFile_LdpcaEncoder.h"

//修改自小小白
unsigned char* BitToByte(unsigned char *Bits, int BitLength, int *numBytes)
{	
	int i, remainder;
	int ByteLength;
	unsigned char *Bytes;
	
	ByteLength = BitLength >> 3; //BitLength / 8
	remainder = BitLength & 7;	//BitLength % 8
	if(remainder != 0)
		ByteLength++;

	Bytes = (unsigned char *)malloc(sizeof(unsigned char) * ByteLength);	
	*numBytes = ByteLength;	//output number of bytes

	int byteIdx = 0;
	unsigned char tmp = 0;
	for(i=0; i<BitLength; i++){
		tmp |= Bits[i];
		if(((i+1)&7) == 0){	//(i+1)%8 == 0
			Bytes[byteIdx++] = tmp;
			tmp = 0;
		}
		else
			tmp <<= 1;
	}
	if(remainder != 0)
		Bytes[byteIdx++] = tmp<<(7-remainder);

	return Bytes;
}

unsigned char* ByteToBit(unsigned char *Bytes, int ByteLength, int SourceLength)
{
	unsigned char *Bits, tmp;
	int i;
	
	if((ByteLength<<3) < SourceLength)
		errorMsg("In function ByteToBit: SourceLength is larger than ByteLength");

	Bits = (unsigned char *)malloc(sizeof(unsigned char) * SourceLength);

	int byteIdx = 0;
	for(i=0; i<SourceLength; i++){
		if((i&7) == 0)	//i%8 == 0
			tmp = Bytes[byteIdx++];
		else
			tmp <<= 1;
		Bits[i] = tmp >> 7;
	}

	return Bits;
}

//小小白
unsigned char* LdpcaEncoder(unsigned char *source, int* SyndromeByteLength, unsigned char *crc)
{    
    int k, l, numBytes;
	unsigned char *source_Bytes, *accumulatedSyndrome, *accuSyn_Byte;

	//trans to Bytes
	source_Bytes = BitToByte(source, EnCodeLength, &numBytes);
			
	//output CRC
	*crc = CRC8(source_Bytes, numBytes);
	free(source_Bytes);

    accumulatedSyndrome = (unsigned char *)calloc(EnCodeLength, sizeof(unsigned char));	 
    
    //source*H' (= H * source)
    for(k=0; k<EnCodeLength; k++)
        for(l=EpJC[k]; l<EpJC[k+1]; l++)
            accumulatedSyndrome[EpIrRead[l]] ^= source[k];
    
    //accumulate
    for(k=1; k<EnCodeLength; k++)
        accumulatedSyndrome[k] ^= accumulatedSyndrome[k-1];

	//trans accumulatedSyndrome(bit) to Bytes
	accuSyn_Byte = BitToByte(accumulatedSyndrome, EnCodeLength, &numBytes);
	free(accumulatedSyndrome);
	
	//output Number of bytes of syndrome
	*SyndromeByteLength = numBytes;

	return accuSyn_Byte;   
}
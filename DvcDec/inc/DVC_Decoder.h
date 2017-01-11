#ifndef DVC_DECODER_H
#define DVC_DECODER_H

//decode frames in the same GOP by hierachical order
void decodeGOP(int begin, int end);

//skip frames(intra block bitstream) in the same GOP by hierachical order
void skipGOP(int begin, int end);

#endif
#ifndef DEBLOCK_H
#define DEBLOCK_H

void init_type_map(unsigned char *type_map, unsigned char *intraBlockFlag, unsigned char *skipBlockFlag);
void deblock(unsigned char *deblockimg, // deblocking frames
        unsigned char *recon,      // pixel-domain recon frame            
        int *trans_sideImg,
        int *trans_reconImg,
        unsigned char *type_map,
        int begin);

#endif
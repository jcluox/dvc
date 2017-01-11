#ifndef DEFINES_H
#define DEFINES_H

#define BLOCKSIZE44 16
#define BLOCKSIZE88 64

#define PARAMETERNUM 12	//number of parameters in config file
#define ET_SIZE 300

#define SKIPTHRESHOLD 10.0
#define U 3.0	//parameter of number of skip/intra block
//#define U_SKIP 2.5	//parameter of number of skip block
//#define U_INTRA 3.0	//parameter of number of intra block
//#define SKIPMIN 15	//minimum number of skip blocks of skip mode

#define GAMA 2.0	//block mode selection (for rate estimation)


#define MAX_QP 51
#define Q_BITS 15
#define MAX_IMGPEL_VALUE 255


#define ROUND(x) ( (x) >= 0.0 ? (int)((x)+0.5) : -(int)(-(x)+0.5) )

#ifdef LINUX
#define STRCASECMP strcasecmp
#else
#define STRCASECMP _strcmpi
#endif

#endif

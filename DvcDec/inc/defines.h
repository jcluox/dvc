#define BLOCKSIZE44 16
#define BLOCKSIZE88 64
#define BLOCKSIZE1616 256

#define PARAMETERNUM 10	//number of parameters in config file
#define ET_SIZE 300

#define SEARCHRANGE 32
#define ME_RANGE ((SEARCHRANGE*2)+1)*((SEARCHRANGE*2)+1)
#define K 0.05f	//use for cost function of motion estimation
#define LEARNSEARCHRANGE 4 //use for motion learning
#define ML_RANGE ((LEARNSEARCHRANGE*2)+1)*((LEARNSEARCHRANGE*2)+1)
#define MU 0.01f //use for motion learning

#define WZ_B 0
#define INTRA_B	1
#define SKIP_B 2

#define MAX_QP 51
#define MAX_IMGPEL_VALUE 255


#define ROUND(x) ( (x) >= 0.0 ? (int)((x)+0.5) : -(int)(-(x)+0.5) )

#ifndef _WIN32
#define STRCASECMP strcasecmp
#else
#define STRCASECMP _strcmpi
#endif


//boris
#define SQR_FILTER 36
#define FILTER_SIZE 6
#define IMG_PAD_SIZE 4
#define FILTER_OFFSET 2
//#define max(a, b) (((a) > (b)) ? (a) : (b))
//#define min(a, b) (((a) < (b)) ? (a) : (b))

//show time information of each bottleneck component in decoding process
#define SHOW_TIME_INFO
#ifdef SHOW_TIME_INFO
	//#define LDPC_ANALYSIS   //計算LDPC每一個步驟的時間，會加上cudaThreadSynchronize() 故執行時間較慢
#endif

//rate control test
//#define DYNAMICSTEP

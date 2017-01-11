#include <stdio.h>
#include <stdlib.h>
#include "error.h"

char errorText[ET_SIZE];

void errorMsg(const char *msg){
	printf("error:%s\n", msg);
#ifdef _WIN32
	system("pause");
#endif
	exit(-1);
}

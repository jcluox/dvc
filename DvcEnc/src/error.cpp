#include <stdio.h>
#include <stdlib.h>
#include "defines.h"
#include "error.h"

char errorText[ET_SIZE];

void errorMsg(const char *msg){
	printf("error:%s\n", msg);
#ifndef LINUX
	system("pause");
#endif
	exit(-1);
}
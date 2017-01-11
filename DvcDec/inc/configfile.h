#ifndef CONFIGFILE_H
#define CONFIGFILE_H

void displayDecodeParamters();
int parameterNameToMapIndex(const char * str);
void overrideParameter(char *cmd, char *argFlag);
void readConfigure(int argc, char **argv);

#endif

WORK_DIR=./
BINDIR=$(WORK_DIR)../release/DvcEnc/
BINNAME=DvcEnc

SRCDIR=$(WORK_DIR)src/
OBJDIR=$(WORK_DIR)Release/
INCLUDE_FLAGS=-I$(WORK_DIR)inc/

SRCS=$(shell echo $(SRCDIR)*.cpp)
OBJS=$(SRCS:$(SRCDIR)%.cpp=$(OBJDIR)%.o)
CC=g++

CC_FLAGS=-DLINUX -O2


all: $(BINDIR)$(BINNAME)

$(BINDIR)$(BINNAME): $(OBJS)
	$(CC) -o $@ $(OBJS)

$(OBJDIR)%.o: $(SRCDIR)%.cpp
	$(CC) -c $< -o $@ $(CC_FLAGS) $(INCLUDE_FLAGS)

clean:
	rm -rf $(OBJDIR)*.o
	rm -rf $(BINDIR)$(BINNAME)

WORK_DIR=./
BINDIR=$(WORK_DIR)../release/DvcDec/
BINNAME=DvcDec

SRCDIR=$(WORK_DIR)src/
OBJDIR=$(WORK_DIR)Release/
INCLUDE_FLAGS=-I$(WORK_DIR)inc/

CUDA_INSTALL_PATH=/usr/local/cuda

CUDA_INSTALLED=$(shell test -e $(CUDA_INSTALL_PATH) && echo "1")
ifeq ($(CUDA_INSTALLED),1)
    #GENCODE_SM := -gencode=arch=compute_10,code=\"sm_10,compute_10\"
    #GENCODE_SM := -gencode=arch=compute_20,code=\"sm_20,compute_20\"
    #GENBIN_SM10 := -arch=sm_10
    #GENBIN_SM20 := -arch=sm_20

    ifeq ($(FERMI),1)  #make command parameter
        GENCODE_SM = -gencode=arch=compute_20,code=\"sm_20,compute_20\"
        GENBIN_SM = -arch=sm_20
        CUDACC_FLAGS = -DFERMI
    else
        GENCODE_SM = -gencode=arch=compute_10,code=\"sm_10,compute_10\"
        GENBIN_SM = -arch=sm_10
    endif

    NVCC=nvcc
    CCFILES=$(shell echo $(SRCDIR)*.cpp)
    CUBINFILES=$(SRCDIR)LDPCA_kernel.cu
    #PTXFILES=$(SRCDIR)LDPCA_kernel.cu
    CUFILES=$(shell  find $(SRCDIR) -type f -name "*.cu" | grep -v $(CUBINFILES))
    #CUFILES=$(shell echo $(SRCDIR)*.cu)
    OBJS=$(CCFILES:$(SRCDIR)%.cpp=$(OBJDIR)%.o)
    OBJS+=$(CUFILES:$(SRCDIR)%.cu=$(OBJDIR)%.o)
    ifeq ($(FERMI),1)
        CUBINS=$(CUBINFILES:$(SRCDIR)%.cu=$(BINDIR)%_sm20.cubin)
    else
        CUBINS=$(CUBINFILES:$(SRCDIR)%.cu=$(BINDIR)%_sm10.cubin)
    endif
    #PTXBINS=$(PTXFILES:$(SRCDIR)%.cu=$(BINDIR)%.ptx)
    CUDACC_FLAGS+=-O2 -use_fast_math -Xcompiler -fopenmp --ptxas-options=-v

all: $(BINDIR)$(BINNAME)

$(BINDIR)$(BINNAME): $(OBJS) $(CUBINS) 
	$(NVCC) -o $@ $(OBJS) -lgomp -lcuda

$(OBJDIR)%.o: $(SRCDIR)%.cpp
	$(NVCC) -c $< -o $@ $(CUDACC_FLAGS) -DCUDA $(INCLUDE_FLAGS)

$(OBJDIR)%.o: $(SRCDIR)%.cu
	$(NVCC) -c $< -o $@ $(CUDACC_FLAGS) $(GENCODE_SM)  $(INCLUDE_FLAGS)	

$(BINDIR)%_sm10.cubin: $(SRCDIR)%.cu
	$(NVCC) $(GENBIN_SM) $(INCLUDE_FLAGS) $(CUDACC_FLAGS) -o $@ -cubin $<

$(BINDIR)%_sm20.cubin: $(SRCDIR)%.cu
	$(NVCC) $(GENBIN_SM) $(INCLUDE_FLAGS) $(CUDACC_FLAGS) -o $@ -cubin $<

#$(BINDIR)%.ptx: $(SRCDIR)%.cu
#	$(NVCC) $(GENCODE_SM) $(INCLUDE_FLAGS) $(CUDACC_FLAGS) -o $@ -ptx $<

else
	CC=g++
	SRCS=$(shell echo $(SRCDIR)*.cpp)
	OBJS=$(SRCS:$(SRCDIR)%.cpp=$(OBJDIR)%.o)
	CC_FLAGS=-O2 -fopenmp 


all: $(BINDIR)$(BINNAME)

$(BINDIR)$(BINNAME): $(OBJS)
	$(CC) -o $@ $(OBJS) -fopenmp 

$(OBJDIR)%.o: $(SRCDIR)%.cpp
	$(CC) -c $< -o $@ $(CC_FLAGS) $(INCLUDE_FLAGS)
endif


clean:
	rm -rf $(OBJDIR)*.o
	rm -rf $(BINDIR)$(BINNAME)
	rm -rf $(CUBINS) $(PTXBINS)



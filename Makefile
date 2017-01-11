all:
	make -C DvcEnc/
	make -C DvcDec/
clean:
	make clean -C DvcEnc/
	make clean -C DvcDec/

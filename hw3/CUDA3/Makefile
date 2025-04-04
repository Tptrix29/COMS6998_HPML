SDK_INSTALL_PATH :=  /usr/local/cuda
NVCC=$(SDK_INSTALL_PATH)/bin/nvcc
LIB	   :=  -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudnn -lcuda
OPTIONS   :=  -O3 
#--maxrregcount=100 --ptxas-options -v 

TAR_FILE_NAME  := YourNameCUDA1.tar
EXECS :=  c1 c2 c3
all:$(EXECS)

#######################################################################
clean:
	rm -f $(EXECS) *.o

#######################################################################
c1.o: c1.cu
	${NVCC} $< -c -o $@ $(OPTIONS)
c1: c1.o
	${NVCC} c1.o -o $@ $(LIB) $(OPTIONS)

#######################################################################

c2.o: c2.cu
	${NVCC} $< -c -o $@ $(OPTIONS)
c2: c2.o
	${NVCC} c2.o -o $@ $(LIB) $(OPTIONS)

#######################################################################

c3.o: c3.cu
	${NVCC} $< -c -o $@ $(OPTIONS)
c3: c3.o
	${NVCC} c3.o -o $@ $(LIB) $(OPTIONS)

#######################################################################

	
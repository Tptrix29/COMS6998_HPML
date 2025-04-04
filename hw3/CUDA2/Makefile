CXX=g++
CXXFLAGS=-O3 -Wall -std=c++11

SDK_INSTALL_PATH :=  /usr/local/cuda
NVCC=$(SDK_INSTALL_PATH)/bin/nvcc
LIB	   :=  -L$(SDK_INSTALL_PATH)/lib64 -L$(SDK_INSTALL_PATH)/samples/common/lib/linux/x86_64
OPTIONS   :=  -O3 
#--maxrregcount=100 --ptxas-options -v 

TAR_FILE_NAME  := YourNameCUDA1.tar
EXECS :=  q1 q2 q3
all:$(EXECS)

#######################################################################
clean:
	rm -f $(EXECS) *.o

#######################################################################
q1:
	$(CXX) $(CXXFLAGS) -o q1 q1.cpp

#######################################################################

q2.o : q2.cu
	${NVCC} $< -c -o $@ $(OPTIONS)

q2 : q2.o
	${NVCC} q2.o -o $@ $(LIB) $(OPTIONS)

#######################################################################

q3.o : q3.cu
	${NVCC} $< -c -o $@ $(OPTIONS)
q3 : q3.o
	${NVCC} q3.o -o $@ $(LIB) $(OPTIONS)

#######################################################################

	
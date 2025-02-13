CC=gcc
CFLAGS=-O3 -Wall
MKL_PATH=/opt/intel/oneapi/mkl/2022.2.0

all: dp1 dp2 dp3

dp1: dp1.c
	$(CC) $(CFLAGS) -o dp1 dp1.c

dp2: dp2.c
	$(CC) $(CFLAGS) -o dp2 dp2.c

dp3: dp3.c
	$(CC) $(CFLAGS) -I $(MKL_PATH)/include -o dp3 dp3.c -L $(MKL_PATH)/lib -lmkl_rt

clean:
	rm -f dp1 dp2 dp3 *.o

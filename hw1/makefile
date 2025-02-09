# Define the compiler
CC=gcc

# Define any compile-time flags
CFLAGS=-O3 -Wall

# Define the names of the executables to produce
TARGET1=dp1
TARGET2=dp2

# Define the source files for each target
SRC1=dp1.c
SRC2=dp2.c

# Define the object files for each source
OBJ1=$(SRC1:.c=.o)
OBJ2=$(SRC2:.c=.o)

.PHONY: all clean

all: $(TARGET1) $(TARGET2)

$(TARGET1): $(OBJ1)
	$(CC) $(CFLAGS) -o $(TARGET1) $(OBJ1)

$(TARGET2): $(OBJ2)
	$(CC) $(CFLAGS) -o $(TARGET2) $(OBJ2)

# Define a rule for compiling .c files to .o files
.c.o:
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET1) $(TARGET2) $(OBJ1) $(OBJ2)

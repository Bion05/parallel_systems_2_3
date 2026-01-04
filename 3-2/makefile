# Compiler and flags
MPICC = mpicc
CFLAGS = -Wall -O2
TARGET = mpi_sparse_mv
SRC = mpi_sparse_mv.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(MPICC) $(CFLAGS) $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET)

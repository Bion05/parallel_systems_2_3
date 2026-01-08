# Compiler and flags
MPICC = mpicc
CFLAGS = -Wall -O2 -fopenmp   # <-- προσθέτουμε -fopenmp
TARGET = hybrid_sparse_mv
SRC = hybrid_sparse_mv.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(MPICC) $(CFLAGS) $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET)

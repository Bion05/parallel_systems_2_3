# Compiler and flags
MPICC = mpicc
CFLAGS = -Wall -O2
TARGET = mpi_poly_mult
SRC = mpi_poly_mult.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(MPICC) $(CFLAGS) $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET)

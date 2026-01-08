# Compiler and flags
CC = gcc
CFLAGS = -Wall -O2 -fopenmp
TARGET = sparse_mv
SRC = sparse_mv.c

# Default target
all: $(TARGET)

# Build executable
$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o $(TARGET)

# Clean build artifacts
clean:
	rm -f $(TARGET)

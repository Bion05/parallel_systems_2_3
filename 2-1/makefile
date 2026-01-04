# Compiler and flags
CC = gcc
CFLAGS = -Wall -O2 -fopenmp
TARGET = poly_mult
SRC = poly_mult.c

# Default target
all: $(TARGET)

# Build executable
$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o $(TARGET)

# Clean build artifacts
clean:
	rm -f $(TARGET)

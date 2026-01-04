# Compiler and flags
CC = gcc
CFLAGS = -Wall -O2 -fopenmp
TARGET = mergesort
SRC = mergesort.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET)

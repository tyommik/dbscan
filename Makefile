# Makefile for C++ project
CC = g++
CFLAGS = -I vendor/ -std=c++2a
SRC = example.cpp dbscan.cpp
OBJ = $(SRC:.cpp=.o)
OUT = example

all: $(OUT)

$(OUT): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^

%.o: %.cpp
	$(CC) $(CFLAGS) -c -o $@ $<

.PHONY: clean

clean:
	rm -f $(OBJ) $(OUT)


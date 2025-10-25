CC=gcc

Cflags=-Wall -Wextra -Wpedantic -O2 -fPIC -shared

all:
	$(CC) $(Cflags) -o loader.so loader.c
 
clean:
	rm loader.so

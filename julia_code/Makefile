CC=clang

Cflags=-Wall -Wextra -Wpedantic -O2 -fPIC -shared

all:
	$(CC) $(Cflags) -o loader.so loader.c -lpthread
 
clean:
	rm loader.so

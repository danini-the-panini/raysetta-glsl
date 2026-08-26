CC=cc
CFLAGS=-g -Wall -Wextra $(shell pkg-config --cflags glfw3)
LDFLAGS=$(shell pkg-config --static --libs glfw3)

main: main.c
	$(CC) $(CFLAGS) -o main main.c $(LDFLAGS)

clean:
	rm main

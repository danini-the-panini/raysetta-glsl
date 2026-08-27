CC=cc
CFLAGS=-g -Wall -Wextra $(shell pkg-config --cflags glfw3)
LDFLAGS=$(shell pkg-config --static --libs glfw3)

main: main.c gl.h linmath.h camera.h stb_image.h
	$(CC) $(CFLAGS) -o main main.c $(LDFLAGS)

clean:
	rm main

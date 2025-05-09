file_name = main

# Compiler
CC = g++

all:
	$(CC) -o $(file_name) $(file_name).cpp

run: all
	reset
	./$(file_name)

clean:
	rm -f $(file_name)


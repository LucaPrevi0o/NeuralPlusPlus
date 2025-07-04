file_name = test

# Compiler - Use Homebrew's g++ or clang++
CC = clang++

# Compiler flags
CXXFLAGS = -std=c++20 -Wall -Wextra -O2
MAKEFLAGS += --no-print-directory

# Include directory
INCLUDES = -Iinclude

all:
	@$(CC) $(CXXFLAGS) $(INCLUDES) -o $(file_name) $(file_name).cpp

run: all
	@reset
	@./$(file_name)

clean:
	@rm -f $(file_name)

.PHONY: all run clean


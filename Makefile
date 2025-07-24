name = test

# Compiler - Use Homebrew's g++ or clang++
CC = clang++

# Compiler flags
CXXFLAGS = -std=c++20 -Wall -Wextra -O2
MAKEFLAGS += --no-print-directory

# Include directory
INCLUDES = -Iinclude

all:
	@$(CC) $(CXXFLAGS) $(INCLUDES) -o $(name) $(name).cpp

run: all
	@reset
	@./$(name)

clean:
	@rm -f $(name)

.PHONY: all run clean


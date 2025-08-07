name = test

# Compiler - Use Homebrew's g++ or clang++
CC = clang++

# Compiler flags
CXXFLAGS = -std=c++20 -Wall -Wextra -O2
MAKEFLAGS += --no-print-directory

# Include directory
INCLUDES = -Iinclude

compile:
	@$(CC) $(CXXFLAGS) $(INCLUDES) -o $(name) $(name).cpp

run: compile
	@./$(name)

all: compile run

clean:
	@rm -f $(name)

.PHONY: all compile run clean


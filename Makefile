target = main

# Compiler - Use Homebrew's g++ or clang++
CC = clang++

# Compiler flags
CXXFLAGS = -std=c++20 -Wall -Wextra -O2
MAKEFLAGS += --no-print-directory

# Include directory
INCLUDES = -Iinclude

compile:
	@$(CC) $(CXXFLAGS) $(INCLUDES) -o $(target) $(target).cpp

run:
	@./$(target)

all: compile run

clean:
	@rm -f $(target)

help:
	@echo make [option] [target]
	@echo "Available options:"
	@echo "- compile    Compile the project (default name: main)"
	@echo "- run        Run the project (default name: main)"
	@echo "- all        Compile and run the project (default name: main)"
	@echo "- clean      Remove the binary file (default name: main)"
	@echo "- help       Show this help message"

.PHONY: all compile run clean help
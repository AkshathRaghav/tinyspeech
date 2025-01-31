$(VERBOSE).SILENT:
.DEFAULT_GOAL := help
SHELL := /bin/bash


CC = gcc
CFLAGS = -g -lm -Wall -I$(INFERENCE_DIR)/include -I$(VERIFICATION_DIR)/include

# Paths
SOURCE_FOLDER = /depot/euge/data/araviki/vsdsquadronmini
VERIFICATION_DIR = $(SOURCE_FOLDER)/verification
INFERENCE_DIR = $(SOURCE_FOLDER)/inference
SUBFOLDERS := $(wildcard $(VERIFICATION_DIR)/*)
LAYER_NAMES := $(notdir $(SUBFOLDERS))

SOURCES = $(INFERENCE_DIR)/source/misc.c \
          $(INFERENCE_DIR)/source/modules.c \
          $(INFERENCE_DIR)/source/tensor.c \
          $(INFERENCE_DIR)/source/tinyspeech.c
OBJECTS = $(SOURCES:.c=.o)

# Targets
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

.DEFAULT_GOAL := debug 

debug: build_engine infer

build_engine:
	@echo "Building inference engine..."
	$(CC) $(CFLAGS) $(wildcard $(INFERENCE_DIR)/src/*.c) -o inference_engine
	@echo "Completed at ./inference_engine!"

# Make split terminals
# "make infer" on one 
# "gdb ./inference_engine" then "target remote | vgdb --pid=" iwthin gdb 
# gdb "detach, quit" will work, but valgrind won't stop
# ps aux | grep valgrind | kill -9 on the valgrind terminal to stop running valgrind processes 

## Ugly, please find a fix ^^ 
infer:
	@echo "Running inference engine with Valgrind..."
	valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes --verbose --vgdb=yes --vgdb-error=0 ./inference_engine

verify_all: 
	@$(MAKE) venv
	@$(foreach LAYER,$(LAYER_NAMES), \
		cd $(VERIFICATION_DIR)/$(LAYER) && \
		echo "Running tests in $(LAYER)" && \
		python3 test.py && \
		$(CC) test.c -o test $(CFLAGS) && \
		valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes --verbose --log-file=valgrind-$(LAYER).txt ./test;)
	@$(MAKE) clean

verify:
	@if [ -z "$(LAYER)" ]; then \
		echo "Error: LAYER variable is not set. Please specify a layer to verify."; \
		exit 1; \
	fi
	@$(MAKE) venv
	@cd $(VERIFICATION_DIR)/$(LAYER) && \
	python3 test.py && \
	$(CC) test.c -o $(LAYER) $(CFLAGS) && \
	valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes --verbose --log-file=valgrind-out1.txt ./$(LAYER) && \
	@cd ../..

venv:
	@echo "Setting up virtual environment..."
	python3 -m venv ./.venv
	. ./.venv/bin/activate

.PHONY: clean
clean:
	@$(foreach DIR, $(SUBFOLDERS), \
		echo "Cleaning directory: $(DIR)"; \
		rm -f $(DIR)/*.bin $(DIR)/*.txt $(DIR)/*.o $(DIR)/*.out $(DIR)/test;)
	@rm -f inference_engine

.PHONY: lint
lint:
	echo "isort:"
	echo "======"
	python3 -m isort --profile=black --line-length=120 .
	echo
	echo "black:"
	echo "======"
	python3 -m black --line-length=120 .
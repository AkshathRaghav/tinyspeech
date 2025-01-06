$(VERBOSE).SILENT:
.DEFAULT_GOAL := help
SHELL := /bin/bash


CC = gcc
CFLAGS = -lm -Wall -Iverification
	
	
# Paths
VERIFICATION_DIR = /depot/euge/data/araviki/vsdsquadronmini/verification
SUBFOLDERS := $(wildcard $(VERIFICATION_DIR)/*)
LAYER_NAMES := $(notdir $(SUBFOLDERS))

# Targets
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

venv: . ./.venv/bin/activate.csh


.PHONY: clean
clean:
	@$(foreach DIR, $(SUBFOLDERS), \
		echo "Cleaning directory: $(DIR)"; \
		rm -f $(DIR)/*.bin $(DIR)/*.txt $(DIR)/*.o $(DIR)/*.out $(DIR)/test;)

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



.PHONY: lint
lint:
	echo "isort:"
	echo "======"
	python3 -m isort --profile=black --line-length=120 .
	echo
	echo "black:"
	echo "======"
	python3 -m black --line-length=120 .
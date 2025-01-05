$(VERBOSE).SILENT:
.DEFAULT_GOAL := help
SHELL := /bin/bash


CC = gcc
CFLAGS = -lm -Wall -Iverification


# Paths
VERIFICATION_DIR = verification

# Targets
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

venv: . ./.venv/bin/activate.csh

clean:
	@find $(VERIFICATION_DIR) -name "*.o" -delete
	@find $(VERIFICATION_DIR) -name "$(LAYER)" -delete
	@find $(VERIFICATION_DIR) -name "valgrind-out1.txt" -delete

verify:
	@if [ -z "$(LAYER)" ]; then \
		echo "Error: LAYER variable is not set. Please specify a layer to verify."; \
		exit 1; \
	fi
	@$(MAKE) venv
	@cd $(VERIFICATION_DIR)/$(LAYER) && \
	python3 test.py && \
	$(CC) test.c -o $(LAYER) $(CFLAGS) && \
	valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes --verbose --log-file=valgrind-out1.txt ./$(LAYER)


.PHONY: lint
lint:
	echo "isort:"
	echo "======"
	python3 -m isort --profile=black --line-length=120 .
	echo
	echo "black:"
	echo "======"
	python3 -m black --line-length=120 .
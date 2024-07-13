$(VERBOSE).SILENT:
.DEFAULT_GOAL := help

.PHONY: lint
lint:
	echo "isort:"
	echo "======"
	python3 -m isort --profile=black --line-length=120 .
	echo
	echo "black:"
	echo "======"
	python3 -m black --line-length=120 .

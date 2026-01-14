.PHONY: install-dev clean format check install uninstall test diff-test


install-dev:
	python3 -m pip install -U pip
	python3 -m pip install -e . --group dev

clean:
	rm -rf build
	rm -rf dist
	rm -rf *.egg-info
	find . -name "*pycache*" | xargs rm -rf

format:
	ruff format mahjax
	blackdoc mahjax
	ruff check mahjax --fix

check:
	ruff format mahjax --check
	blackdoc mahjax --check
	ruff check mahjax
	mypy mahjax

install:
	python3 -m pip install -U pip setuptools
	python3 -m pip install .

uninstall:
	python3 -m pip uninstall mahjax -y

test:
	python3 -m pytest -n 4 -vv tests --doctest-modules mahjax --ignore mahjax/experimental

test-with-codecov:
	python3 -m pytest -n 4 -vv tests --doctest-modules mahjax --ignore mahjax/experimental --cov=mahjax --cov-report=term-missing --cov-report=html

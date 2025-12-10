.PHONY: install-dev clean format check install uninstall test diff-test


install-dev:
	python3 -m pip install -U pip
	python3 -m pip install -r requirements/requirements-dev.txt

clean:
	rm -rf build
	rm -rf dist
	rm -rf *.egg-info
	find . -name "*pycache*" | xargs rm -rf

format:
	black mahjax
	blackdoc mahjax
	isort mahjax

check:
	black mahjax --check --diff
	blackdoc mahjax --check
	flake8 --config pyproject.toml --ignore E203,E501,W503,E704,E741 mahjax
	mypy --config pyproject.toml mahjax  --ignore-missing-imports
	isort mahjax --check --diff

install:
	python3 -m pip install -U pip setuptools
	python3 -m pip install .

uninstall:
	python3 -m pip uninstall mahjax -y

test:
	python3 -m pytest -n 4 -vv tests --doctest-modules mahjax --ignore mahjax/experimental

test-with-codecov:
	python3 -m pytest -n 4 -vv tests --doctest-modules mahjax --ignore mahjax/experimental --cov=mahjax --cov-report=term-missing --cov-report=html

# Supervised Medical

## Train

`python3 -m src.train`

## Test

`python3 -m pytest`

## Requirements

Requires Python >=3.8

## Contribute

Install pre-commit hooks with: `pre-commit install`

This runs pre-commit hooks, including `black` to ensure code style consistency.

The same `black` hook can also be run independently with: `python3 -m black . --line-length 120`

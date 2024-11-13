# Getting started with testings

We use pytest in order to run the unit and integration testings.

Pytest takes the following argument flags:

`-m`: Runs tests matching the given mark.

`-k`: Runs tests that match the substring given.

`-v`: Increase verbosity.

`-x`: Exit on test failed.

`-s`: No capture.

It is recommended to do a `dvc repro -f` before running the test to generate the necessary files.

## Unit tests

In order to run all unit tests we need to use the following command:
```python
pytest -m "not integtest" -v -s -x
```

If we want to run a specific phase of the process we can run the following:

Load data:
```python
pytest -k load -v -s
```
Preprocess data:
```python
pytest -k preprocess -v -s
```
Model parameters:
```python
pytest -k train -v -s
```
Model evaluation:
```python
pytest -k evaluation -v -s
```
Model input:
```python
pytest -k input -v -s
```
Inferences:
```python
pytest -k inference -v -s
```

# Integration tests
In order to run the integration test we need to use the following command:
```python
pytest -m integtest -v -s
```
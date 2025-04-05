.PHONY: outliers lstm all

outliers:
	python outliers.py

lstm:
	python lstm.py

all:
	python outliers.py
	python lstm.py
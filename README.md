
# TC5044 MLOPS - Team 25
# Overview
### The opportunity presented in this document lies in the lack of an adequate system for the classification and selection of apartments in the United States of America. However, a detailed study is available that includes key apartment characteristics such as location, price, size, among others, allowing for a more in-depth and structured analysis. The objective is to develop a model that enables the visualization of the best price for renting an apartment based on the characteristics from the dataset available.
# The dataset used in this project is Apartment for rent classified https://archive.ics.uci.edu/dataset/555/apartment+for+rent+classified

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

MLOps project

## Project Organization
```
MLOpsTeam25/
├── .dvc
├── data
│   ├── interim
│   ├── processed
│   └── raw
│   └── raw_full
├── docs
├── mlops
├── models
    ├── ApartmentRent.py
    ├── __init__.py
    ├── evaluate.py
    ├── load_data.py
    ├── preprocess_data.py
    ├── reproduce.py
    ├── train.py
    ├── utils.py
├── notebooks
├── references
├── reports
│   └── figures
├── requirements.txt
├── setup.cfg                     
├── pyproject.toml  
├── params.yaml
├── README.md

```

## Instructions to reproduce the pipeline

If you are using a local version of MLflow, use the following command:

1. ```mlflow server --host 127.0.0.1 --port 5000```

To reproduce this project use the following instructions:

1. Install all the dependencies ```pip install -r requirements.txt```
2. Execute the pipeline ```dvc repro --force```

 
--------![image](https://github.com/user-attachments/assets/b6155c2c-f3e2-4ddb-b2dc-105e201e5b69)

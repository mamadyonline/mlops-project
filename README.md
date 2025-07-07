# MLOps project

## Context
This project is the final project for MLOps zoomcamp by DataTalks

## Goal of the project
Build an end-to-end ML pipeline, from data selection to model building, deployment and monitoring, all of this by following good software engineering practices.

## Problem description
Cardiovascular diseases are one of the leading causes of death globally, accounting for millions of fatalities each year. Early detection and risk assessment are crucial to prevent severe outcomes. This project aims to build an end-to-end machine learning pipeline for the detection of cardiovascular disease risk based on patient health data.

The pipeline automates the entire ML lifecycle : from data collection and preprocessing to model training, evaluation, and deployment, while ensuring reproducibility, scalability, and monitoring. By leveraging MLOps best practices, the project demonstrates how to operationalize a machine learning model in a real-world healthcare scenario.

### Dataset
We will use the [UCI heart disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease). It contains 13 main features (numerical + categorical) plus the target (a binary value indicating risk or no). For a complete description of the dataset, please refer to the *UCI* website.

### Stack
* uv for dependency management
* xgboost for classification model
* mlflow for model tracking
* apache airflow for model orchestration
* evidently for model monitoring

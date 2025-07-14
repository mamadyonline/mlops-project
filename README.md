# MLOps project

## Context
This project is the final project for MLOps zoomcamp by DataTalks

## Goal of the project
Build an end-to-end ML pipeline, from data selection to model building, deployment and monitoring, all of this by following good software engineering practices.

## Problem description
Cardiovascular diseases are one of the leading causes of death globally, accounting for millions of fatalities each year. Early detection and risk assessment are crucial to prevent severe outcomes. This project aims to build an end-to-end machine learning pipeline for the detection of cardiovascular disease risk based on patient health data.

The pipeline automates the entire ML lifecycle : from data collection and preprocessing to model training, evaluation, and deployment, while ensuring reproducibility, scalability, and monitoring. By leveraging MLOps best practices, the project demonstrates how to operationalize a machine learning model in a real-world healthcare scenario.

### Dataset
We will use the [UCI heart disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease). The dataset version I am using is from this [kaggle link](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset?resource=download&select=heart.csv). It contains 13 main features (numerical + categorical) plus the target (a binary value indicating risk or no). For a complete description of the dataset, please refer to the *UCI* website.

## Tech stack
* uv for dependency management
* xgboost for classification model
* mlflow for model tracking
* apache airflow for model orchestration
* evidently for model monitoring
* ruff for code formatting and linting
* fastAPI for the web service
* pytest for unit testing
* AWS for cloud: ec2 for mlflow server, rds for mlflow db and s3 for artifacts storage
  * Check screenshots in `images` folder

## Usage

### Dependencies
Install the dependencies from `requirements.txt` (This is the full library versions I ended up with. You can install otherwise the minimal dependencies in `requirements.in` and sort out the others on your way). For example with uv:

```bash
uv pip install -r requirements.txt
```

### Model deployment

The easiest way to use the model is through the web service app. The folder `webservice` contains two files `app.py` and `client_example.py`. You can test it by following the steps below:

```bash
cd webservice
# in terminal 1, launch the app
python app.py
# in terminal 2, test it all
python client_example.py
# One predict
curl -X POST http://localhost:8000/predict \             
  -H "Content-Type: application/json" \
  -d '{
    "age": 45,
    "sex": 1,
    "cp": 2,
    "trestbps": 130,
    "chol": 250,
    "fbs": 0,
    "restecg": 1,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 1.5,
    "slope": 1,
    "ca": 1,
    "thal": 2
  }'
```

Note that all the training, monitoring, pipeline orchestration were done with AWS, so testing the full pipeline can be tricky. So I provide a module (`train_model.py`) to train the model and save artifacts locally with mlflow launched locally on port `6000` (you can change it at your own convenience).

### Testing the ML pipeline orchestration

You can install apache airflow by doing the following command:

```bash
uv pip install "apache-airflow[celery]==3.0.2" --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.0.2/constraints-3.9.txt"
```
Afterwards, you will need to specify the dags directory (absolute). For example you can do:

```bash
export AIRFLOW_HOME=~/Desktop/mlops-project/airflow
```
Then you can launch airflow from the project root directory and check on `localhost:8080` to see the dags. 

```bash
airflow standalone
```

Another way is to test specific tasks from each dag. For example, you can do:

```bash
airflow tasks test heart_disease_batch_deployment generate_predictions 2025-07-10
```

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
* optuna for hyperparameter optimization
* mlflow for model tracking
* apache airflow for model orchestration
* evidently for model monitoring
* ruff for code formatting and linting
* fastAPI for the web service
* pytest for unit testing
* AWS for cloud: ec2 for mlflow server, rds for mlflow db and s3 for artifacts storage
  * Check screenshots in `images` folder

## Usage
To use/test the code, first download the project locally:

```bash
git clone git@github.com:mamadyonline/mlops-project.git
```

Then, use `uv` or any other virtual environment management tool to create a venv and all the dependencies.

```bash
uv venv --python 3.9
```

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

Note that all of this was done with AWS setup. So you will need to have pre-configured the three main services (AWS rds, ec2, s3) and set the environment variables `AWS_PROFILE` and `MY_AWS_EC2_SERVER`.

You can install apache airflow by doing the following command:

```bash
uv pip install "apache-airflow[celery]==3.0.2" --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.0.2/constraints-3.9.txt"
```
Afterwards, you will need to specify the dags directory (absolute path). For example you can do:

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

## Images

Since, it may be hard to run all of the steps without doing some modifications, below is a structure walkthrough the results with images. For each of the main step, I list down the corresponding images:

### Global
*  Pipeline orchestration with AIRFLOW homepage (running locally): `images/airflow_homepage.png`
*  Model tracking server with MLFlow homepage (running on an AWS EC2 server): `images/mlflow_web_ui.png`
*  Model tracking database with MLFlow homepage (running on an AWS RDS db): `images/aws_rds_mlflow_db.png`
*  

### Model tracking
* Model tracking artifacts (stored in AWS s3): `images/pipeline_artifacts.png`, `images/aws_s3_mlflow_artifacts.png`
* Model experimentation (stored in AWS s3): `images/mlflow_xp_1.png`, `images/pipeline_artifacts.png`,

### Model deployment
* Model deployment artifacts (stored in AWS s3): `images/deployment_artifacts.png`

### Model monitoring
* Model monitoring reports (stored in AWS s3): `images/monitoring_reports.png`

### Pipeline orchestration
* Training
  * ![Training Dag structure](/images/ml_training_dag_structure.png "Training dag structure")
  * Global view: `images/ml_training_pipeline_airflow.png`
  * Data load stage: `images/ml_training_pipeline_load_data_airflow.png`
  * Hyperparameter optimization: `images/ml_training_pipeline_optimize_airflow.png`
  * Model training: `images/ml_training_pipeline_train_airflow.png`
* Deployment
  * ![Batch Dag structure](/images/ml_batch_dag_structure.png "Batch deployment dag structure")
  * Global view: `images/ml_batch_deployment_pipeline_airflow.png`
  * Data load stage: `images/ml_batch_load_airflow.png`
  * Predictions: `images/ml_batch_predictions_airflow.png`
  * Performance metrics: `images/ml_batch_metrics_airflow.png`
  * Monitoring: `images/ml_batch_monitoring_airflow.png`

# Cloud-For-AI-Assignment

## Data set
https://www.kaggle.com/datasets/samuelcortinhas/sports-balls-multiclass-image-classification/data

## Application

### Local

Create a new python environment (tested with python 3.10) and install dependencies using

```shell
pip install -r requirements-local.txt
```

#### Frontend
The frontend is built using Streamlit. In order to run it, execute following commands:

```shell
cd <project-root>/app
streamlit run home.py
```

Note: change `base_url = "http://localhost:8000"` in main.py.

#### Backend
The backend is built using Fastapi. In order to run it, execute following commands:

```shell
cd <project-root>/api
uvicorn main:app --host api --port 8000
```


### Docker

#### Frontend
The frontend is built using Streamlit. In order to run it, execute following commands:

```shell
cd <project-root>/app
docker build -t <tag-name> .
docker run -p 8501:8501 <tag-name>
```

Navigate in your browser to http://localhost:8501.

#### Backend
The backend is built using Fastapi. In order to run it, execute following commands:

```shell
cd <project-root>/api
docker build -t <tag-name> .
docker run -p 8000:8000 <tag-name>
```

#### Docker-compose
Run both front- and backend simultaneously with docker compose:

```shell
docker-compose build
docker-compose up
```

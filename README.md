# Cloud-For-AI-Assignment

## Data set
https://www.kaggle.com/datasets/samuelcortinhas/sports-balls-multiclass-image-classification/data

## Application

### Local
- Create Python virtual environment (tested with Python 3.10)
- Install dependencies
  ```shell
    pip install -r requirements.txt
  ```
- Run the streamlit app
  ```shell
    streamlit run app/home.py
  ```

### Container

- Build a Docker image with tag name e.g. cloud-for-ai
  ```shell
    docker build -t cloud-for-ai .
  ```

- Run the Docker image
  ```shell
    docker run -p 8501:8501 cloud-for-ai
  ```

- Navigate in your browser to http://localhost:8501

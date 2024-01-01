FROM tiangolo/uvicorn-gunicorn-fastapi:python3.10

WORKDIR /usr/src/app

COPY . .

RUN apt-get update
RUN apt install -y libgl1-mesa-glx
RUN pip install -r requirements.txt

EXPOSE 8000

# Run app.py when the container launches
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

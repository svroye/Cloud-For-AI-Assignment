FROM tiangolo/uvicorn-gunicorn-fastapi:python3.10

WORKDIR /usr/src/app

COPY . .

EXPOSE 8000

# Run app.py when the container launches
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

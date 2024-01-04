FROM python:3.10.13

WORKDIR /usr/src/app

COPY . .

RUN apt-get update
RUN apt install -y libgl1-mesa-glx
RUN pip install -r requirements.txt

EXPOSE 8501

# Run app.py when the container launches
CMD ["streamlit", "run", "home.py"]

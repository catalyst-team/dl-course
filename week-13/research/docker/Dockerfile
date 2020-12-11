FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip
RUN pip install --trusted-host pypi.python.org -r /app/requirements.txt
RUN apt-get update -y && apt-get install wget unzip -y

COPY . .

ENV DATA_PATH=/app/data
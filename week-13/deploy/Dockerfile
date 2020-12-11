FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN apt-get update -y && \
    apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev wget curl

ENV PYTHONPATH=/src

WORKDIR /src

COPY requirements.txt .
RUN pip install -r requirements.txt --ignore-installed PyYAML

RUN wget https://www.dropbox.com/s/ud4xa1ytl7ea6rg/model.tar && \
    tar -xvf model.tar && \
    rm model.tar && \
    mv model /model

COPY . /src
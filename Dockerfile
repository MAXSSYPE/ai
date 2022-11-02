FROM tensorflow/tensorflow:latest-gpu

WORKDIR /

COPY main.py /main.py

RUN python main.py

FROM mcr.microsoft.com/mirror/docker/library/ubuntu:22.04
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update -qq && \
    apt-get upgrade -y && \
    apt-get install -y gcc build-essential wget python3 python3-pip 
RUN rm -rf /var/lib/apt/lists/*
RUN apt-get clean
RUN pip3 install --upgrade pip
RUN pip3 install poetry pytest
WORKDIR /anomaly-detector
COPY . .
RUN cd anomaly-detector && poetry env use 3.10 && \
    poetry lock --no-update && \
    poetry install
RUN pip3 install Cython numpy
RUN cd ..
RUN python3 setup.py build_ext --inplace
RUN cd anomaly-detector/tests
# FROM mcr.microsoft.com/mirror/docker/library/python:3.9-slim

# ARG DEBIAN_FRONTEND=noninteractive

# RUN apt-get update -qq && \
#     apt-get upgrade -y && \
#     apt-get install -y build-essential

# RUN pip3 install --upgrade pip
# RUN pip3 install poetry pytest

# WORKDIR /anomaly-detector
# COPY . .
# RUN cd anomaly-detector && poetry env use 3.9 && \
#     poetry lock --no-update && \
#     poetry install
# RUN pip3 install Cython numpy
# RUN cd ..
# RUN python3 setup.py build_ext --inplace

# ubuntu:22.04 uvad filled with expectedValue error
FROM mcr.microsoft.com/mirror/docker/library/ubuntu:22.04
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update -qq && \
    apt-get upgrade -y
    
RUN apt-get install -y gcc build-essential wget python3.11 python3-pip 
RUN rm -rf /var/lib/apt/lists/*
RUN apt-get clean
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
# RUN update-alternatives --config python3

RUN pip3 install --upgrade pip
RUN pip3 install poetry pytest
WORKDIR /anomaly-detector
COPY . .
RUN cd anomaly-detector && poetry env use 3.11 && \
    poetry lock --no-update && \
    poetry install
RUN pip3 install Cython numpy
RUN cd ..
RUN python3 setup.py build_ext --inplace
RUN cd anomaly-detector/tests
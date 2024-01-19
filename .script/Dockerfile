FROM mcr.microsoft.com/mirror/docker/library/python:3.11

ARG DEBIAN_FRONTEND=noninteractive

ADD requirements.txt .

RUN apt-get update -qq && \
    apt-get upgrade -y && \
    apt-get install -y r-base r-base-dev --fix-missing && \
    rm -rf /var/lib/apt/lists/*
RUN pip3 install --upgrade pip
RUN pip3 install setuptools --upgrade
RUN pip3 install -r requirements.txt
RUN apt-get clean

WORKDIR /anomaly-detector
COPY . .
RUN python3 setup.py build_ext --inplace
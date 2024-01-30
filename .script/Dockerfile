FROM mcr.microsoft.com/mirror/docker/library/python:3.11-slim

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update -qq && \
    apt-get upgrade -y && \
    apt-get install -y build-essential

RUN pip3 install --upgrade pip
RUN pip3 install poetry pytest

WORKDIR /anomaly-detector
COPY . .
RUN cd anomaly-detector && poetry env use 3.11 && \
    poetry lock --no-update && \
    poetry install
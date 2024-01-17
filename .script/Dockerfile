FROM mcr.microsoft.com/mirror/docker/library/ubuntu:18.04

ARG DEBIAN_FRONTEND=noninteractive

ADD requirements.txt .

RUN apt-get update -qq && \
    apt-get upgrade -y && \
    apt-get install -y python3 python3-pip r-base r-base-dev
RUN pip3 install --upgrade pip
RUN pip3 install setuptools --upgrade
RUN pip3 install -r requirements.txt
RUN apt-get clean

RUN python3 setup.py build_ext --inplace
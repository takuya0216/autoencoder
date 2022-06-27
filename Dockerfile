FROM python:3
USER root

ENV LANG ja_JP.UTF-8
ENV LANGUAGE ja_JP:ja
ENV LC_ALL ja_JP.UTF-8
ENV TZ JST-9

RUN mkdir -p /src_container
COPY requirements.txt /src_container
WORKDIR /src_container

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

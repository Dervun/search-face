FROM ubuntu:20.04

WORKDIR /app

ARG DEBIAN_FRONTEND=noninteractive

COPY . .

RUN apt-get -qq update && mkdir ./logs
RUN apt-get install -qq g++ build-essential cmake python3.8 python3-pip >./logs/docker_build.log

RUN pip3 install --upgrade pip
RUN pip3 install -q pipenv
RUN pipenv --bare sync

CMD ["pipenv", "run", "python", "app.py"]


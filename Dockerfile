FROM python:3.8-slim-buster

ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /app

COPY . .

RUN apt-get -qq update && mkdir ./logs && apt-get install -qq g++ build-essential cmake >./logs/docker_build.log
RUN pip3 install -q pipenv
RUN pipenv --bare sync

CMD ["pipenv", "run", "python", "app.py"]


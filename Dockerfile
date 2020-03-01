FROM python:3.8-slim

RUN apt-get update && apt-get upgrade && apt-get install -y libjpeg-dev libpq-dev zlib1g-dev gcc

COPY requirements.txt ./
RUN pip install --quiet -r requirements.txt

ENV APP_HOME /src
WORKDIR $APP_HOME
COPY . ./

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 wander_io.wsgi
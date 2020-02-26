FROM python:3.6-slim

RUN apt-get update && apt install -y libjpeg-dev libpq-dev zlib1g-dev gcc

ENV APP_HOME /src
WORKDIR $APP_HOME
COPY . ./

RUN pip install -r requirements.txt

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 castijuegos.wsgi
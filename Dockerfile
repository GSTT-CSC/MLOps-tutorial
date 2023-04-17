FROM python:3.9

WORKDIR /project

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential git rsync software-properties-common && \
    rm -rf /var/lib/apt/lists/*

ENV PYTHONPATH="/mlflow/projects/code/:$PYTHONPATH"

COPY . .

# install requirements
RUN python -m pip install --no-cache-dir --upgrade pip &&  \
    python -m pip install --no-cache-dir -r requirements.txt
FROM nvidia/cuda:11.4.0-devel-ubuntu20.04 AS build

WORKDIR /project

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
#ENV PYTHONDONTWRITEBYTECODE=1
#ENV LANG=C.UTF-8

# Make sure we use the virtualenv:
ENV PATH="/opt/venv/bin:$PATH"

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential git rsync software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get install -y --no-install-recommends python3.9 python3-pip python3-dev python3.9-venv && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# install requirements
RUN python3.9 -m venv /opt/venv && \
    python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir wheel && \
    python -m pip install --no-cache-dir -r requirements.txt

FROM nvidia/cuda:11.4.0-runtime-ubuntu20.04 as project

WORKDIR /project

# Extra python env
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
#ENV PYTHONDONTWRITEBYTECODE=1
#ENV LANG=C.UTF-8
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH "${PYTHONPATH}:/project"

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential git rsync software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get install -y gcc && \
    apt-get install -y --no-install-recommends python3.9 python3-pip python3-dev python3.9-venv && \
    rm -rf /var/lib/apt/lists/*

COPY --from=build /opt/venv /opt/venv

COPY . .

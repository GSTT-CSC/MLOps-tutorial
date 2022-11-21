<!-- PROJECT HEADING -->
<br />
<p align="center">
<a href="https://github.com/github_username/repo_name">
    <img src="assets/MOps_template_logo.png" alt="Logo" width="50%">
  </a>
<p align="center">
A framework for AI applications for healthcare
<br />
<br />
<a href="https://github.com/GSTT-CSC/Project_template">View repo</a>
·
<a href="https://github.com/GSTT-CSC/Project_template/issues">Report Bug</a>
·
<a href="https://github.com/GSTT-CSC/Project_template/issues">Request Feature</a>
</p>

# New project template

## Introduction
This repository contains an example project using the [csc-mlops](https://github.com/GSTT-CSC/MLOps.git) development platform. 

The example demonstrates how developers can use the csc-mlops XNAT interface, pytorch lightning, and MONAI, to segment the anterior and posterior hippocampus from a brain MRI data.

Future updates will demonstrate:
  - application packaging
  - performance over demographics 

## Getting started
This example project uses the Hippocampus segmentation dataset from http://medicaldecathlon.com/.


### requirements
Docker and docker compose must be setup on your system, as well as >=python3.9

### Setup XNAT and MLOps servers
```shell
git clone https://github.com/NrgXnat/xnat-docker-compose.git xnat-docker-compose
cd xnat-docker-compose
docker compose up -d --build

cd ..

git clone https://github.com/GSTT-CSC/MLOps.git MLOps
cd MLOps/mlflow_server
docker compose up -d --build

cd ..
docker ps
```

If you view running docker containers with `docker ps` you should now see several entries showing the MLOps and xnat stacks, ensure no services display the "unhealthy" status before continuing.


### Clone this repository and setup virtual environment
```shell
git clone https://github.com/GSTT-CSC/MLOps-tutorial.git MLOps-tutorial
cd MLOps-tutorial
python3 -m venv ../mlops-venv
source ../mlops-venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 


## Contact
For bug reports and feature requests please raise a GitHub issue on this repository.


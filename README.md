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
This repository contains a skeleton project template for use with new projects using the [csc-mlops](https://github.com/GSTT-CSC/MLOps.git) development platform. The template provides a starting point with helper classes and functions to facilitate rapid development and deployment of applications.

## Structure
At a minimum users should use the `Experiment` class and the provided `run_project.py` script to set up their experiment.
This template suggests using pytorch-lightning and MONAI for network configuration and DataModules. 
However, this is not strictly necessary and provided the Dockerfile GPU libraries are adapted and the `run_project` function is used then tracking can be performed with any [MLflow compatible framework](https://mlflow.org/docs/latest/tracking.html#automatic-logging).

## Getting started
This project template makes use of classes and functions provided by the [csc-mlops](https://github.com/GSTT-CSC/MLOps.git) package, installing this to your local environment is easy with pip:

```shell
pip install csc-mlops
```

### steps temp
```shell
python3.9 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

At this point user needs to set up xnat via docker compose and add a new project called "hipposeg"
```
python data/create_xnat_data.py config/local_config.cfg
```

- define datamodule
  - xnat_build_dataset
  - actions to fetch image and label
  - set up local interpreter in pycharm
- add xnat config to train.py and pass to datamodule



### Getting the test data
This example project uses the Hippocampus segmentation dataset from http://medicaldecathlon.com/.

https://drive.google.com/file/d/1RzPB1_bqzQhlWvU-YGvZzhx2omcDh38C/view?usp=share_link


## Contact
For bug reports and feature requests please raise a GitHub issue on this repository.


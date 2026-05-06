# Computer Vision Pipeline (CVP)
Code repository for the BDSC Computer Vision Pipeline project.

## Project Goals
- Develop a flexible architecture for implementing image classification models
- Demonstrate use of the pipeline on sample datasets

## Features


## Tech Stack
The following technologies are utilized in this project:
  - [List of programs/packages]

## Quick Start
This project assumes you have the following software installed:
  - Git
  - A command line interface (CLI)
  - Python (>= 3.14)
    - pipenv (>= 2026.5.2)

### Setup
  - Open a new CLI instance in the desired file location
  - Run `git clone https://github.com/LiamSpillane2/Computer-Vision-Pipeline.git`
  - Run `pipenv install`
    - Run `py -m pipenv install` if pipenv is not added to PATH

##  Repository Structure
```text
Computer-Vision-Pipeline/
├── config/                       # Configuration files
├── data/                         # Dataset storage
│   ├── license_plate_detection/  # License plate dataset
│   └── weld_defects/             # Weld defects dataset
├── models/                       # Files of trained models
├── notebooks/                    # Jupyter notebooks for data exploration
├── src/                          # Main body of source code
│   ├── models/                   # Scripts to train and run models
│   └── preprocessing/            # Scripts to prepare data for model ingestion
├── tests/                        # Unit and integration tests
├── .gitignore                    # gitignore file
├── CODE_OF_CONDUCT.md            # Conduct guidelines
├── Pipfile                       # Pipenv environment configuration file
├── Pipfile.lock                  # Pipenv environment lock file
└── README.md                     # Project description and setup instructions
```